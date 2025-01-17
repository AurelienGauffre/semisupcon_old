
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from semilearn.core import AlgorithmBase, SupConLossWeights
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import DistAlignQueueHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool, concat_all_gather
from pytorch_metric_learning import losses

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix

class CoMatch_Net(nn.Module):
    def __init__(self, base, proj_size=128):
        super(CoMatch_Net, self).__init__()
        self.backbone = base
        self.num_features = base.num_features
        
        self.mlp_proj = nn.Sequential(*[
            nn.Linear(self.num_features, self.num_features),
            nn.ReLU(inplace=False),
            nn.Linear(self.num_features, proj_size)
        ])
        
    def l2norm(self, x, power=2):
        norm = x.pow(power).sum(1, keepdim=True).pow(1. / power)
        out = x.div(norm)
        return out
    
    def forward(self, x, **kwargs):
        dic = self.backbone(x)
        # logits = self.backbone(feat, only_fc=True)
        # feat_proj = self.l2norm(self.mlp_proj(feat))
        return dic

    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher


# TODO: move this to criterions
def comatch_contrastive_loss(feats_x_ulb_s_0, feats_x_ulb_s_1, Q, T=0.2):
    # embedding similarity
    sim = torch.exp(torch.mm(feats_x_ulb_s_0, feats_x_ulb_s_1.t())/ T) 
    sim_probs = sim / sim.sum(1, keepdim=True)
    # contrastive loss
    loss = - (torch.log(sim_probs + 1e-7) * Q).sum(1)
    loss = loss.mean()  
    return loss


@ALGORITHMS.register('comatch_proto')
class CoMatchProto(AlgorithmBase):
    """
        CoMatch algorithm (https://arxiv.org/abs/2011.11183).
        Reference implementation (https://github.com/salesforce/CoMatch/).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
            - contrast_p_cutoff (`float`):
                Confidence threshold for contrastive loss. Samples with similarity lower than a threshold are not connected.
            - queue_batch (`int`, *optional*, default to 128):
                Length of the memory bank to store class probabilities and embeddings of the past weakly augmented samples
            - smoothing_alpha (`float`, *optional*, default to 0.999):
                Weight for a smoothness constraint which encourages taking a similar value as its nearby samples’ class probabilities
            - da_len (`int`, *optional*, default to 256):
                Length of the memory bank for distribution alignment.
            - contrast_loss_ratio (`float`, *optional*, default to 1.0):
                Loss weight for contrastive loss
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # comatch specified arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, 
                  contrast_p_cutoff=args.contrast_p_cutoff, hard_label=args.hard_label, 
                  queue_batch=args.queue_batch, smoothing_alpha=args.smoothing_alpha, da_len=args.da_len)
        self.lambda_c = args.contrast_loss_ratio
        self.supcon_loss = losses.SupConLoss()
        self.supcon_loss_weights = SupConLossWeights()

    def init(self, T, p_cutoff, contrast_p_cutoff, hard_label=True, queue_batch=128, smoothing_alpha=0.999, da_len=256):
        self.T = T 
        self.p_cutoff = p_cutoff
        self.contrast_p_cutoff = contrast_p_cutoff
        self.use_hard_label = hard_label
        self.queue_batch = queue_batch
        self.smoothing_alpha = smoothing_alpha
        self.da_len = da_len

        # TODO: put this part into a hook
        # memory smoothing
        self.queue_size = int(queue_batch * (self.args.uratio + 1) * self.args.batch_size)
        self.queue_feats = torch.zeros(self.queue_size, self.args.proj_size).cuda(self.gpu)
        self.queue_probs = torch.zeros(self.queue_size, self.args.num_classes).cuda(self.gpu)
        self.queue_ptr = 0
        
    def set_hooks(self):
        self.register_hook(
            DistAlignQueueHook(num_classes=self.num_classes, queue_length=self.args.da_len, p_target_type='uniform'), 
            "DistAlignHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def set_model(self):
        model = super().set_model()
        model = CoMatch_Net(model, proj_size=self.args.proj_size)
        return model
    
    def set_ema_model(self):
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model = CoMatch_Net(ema_model, proj_size=self.args.proj_size)
        ema_model.load_state_dict(self.check_prefix_state_dict(self.model.state_dict()))
        return ema_model


    @torch.no_grad()
    def update_bank(self, feats, probs):
        if self.distributed and self.world_size > 1:
            feats = concat_all_gather(feats)
            probs = concat_all_gather(probs)
        # update memory bank
        length = feats.shape[0]
        self.queue_feats[self.queue_ptr:self.queue_ptr + length, :] = feats
        self.queue_probs[self.queue_ptr:self.queue_ptr + length, :] = probs      
        self.queue_ptr = (self.queue_ptr + length) % self.queue_size


    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s_0, x_ulb_s_1, y_ulb):
        num_lb = y_lb.shape[0] 

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s_0, x_ulb_s_1))
                # outputs = self.model(inputs)
                # logits, feats = outputs['logits'], outputs['feat']
                # logits_x_lb, feats_x_lb = logits[:num_lb], feats[:num_lb]
                # logits_x_ulb_w, logits_x_ulb_s_0, _ = logits[num_lb:].chunk(3)
                # feats_x_ulb_w, feats_x_ulb_s_0, feats_x_ulb_s_1 = feats[num_lb:].chunk(3)

                outputs = self.model(inputs, contrastive=True)
                contrastive_x = outputs['contrastive_feats']
                contrastive_x_lb = contrastive_x[:num_lb]
                contrastive_x_ulb_w, contrastive_x_ulb_s_0, contrastive_x_ulb_s_1 = contrastive_x[num_lb:].chunk(3)
                proto_proj = outputs['proto_proj']
            else:
                raise ValueError("SemiSupConProto does not support non-cat mode currently")

            # feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb_w, 'x_ulb_s':[feats_x_ulb_s_0, feats_x_ulb_s_1]}
            feat_dict = {'x_lb': contrastive_x_lb, 'x_ulb_w': contrastive_x_ulb_w,
                         'x_ulb_s': [contrastive_x_ulb_s_0, contrastive_x_ulb_s_1]}
            similarity_to_proto = contrastive_x_ulb_w @ proto_proj.t()
            pseudo_label = torch.argmax(similarity_to_proto, dim=1)
            probs = torch.softmax((similarity_to_proto + 1) / 2 / self.args.pl_temp, dim=1)
            # supervised loss
            mask2 = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs, softmax_x_ulb=False)
            maskbool = mask2.bool()
            # maskbool = torch.max(similarity_to_proto, dim=1)[0] > self.p_cutoff
            mask_sum = maskbool.sum()  # number of samples with high confidence
            #sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            
            with torch.no_grad():
                # logits_x_ulb_w = logits_x_ulb_w.detach()
                # feats_x_lb = feats_x_lb.detach()
                # feats_x_ulb_w = feats_x_ulb_w.detach()

                # probs = torch.softmax(logits_x_ulb_w, dim=1)
                # probs = self.compute_prob(logits_x_ulb_w)
                # distribution alignment
                probs = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs.detach())

                probs_orig = probs.clone()
                # memory-smoothing

                if self.epoch > 0 and self.it > self.queue_batch:
                    # contrastive_x_ulb_w replace feats_x_ulb_w
                    A = torch.exp(torch.mm(contrastive_x_ulb_w, self.queue_feats.t()) / self.T)
                    A = A / A.sum(1,keepdim=True)                    
                    probs = self.smoothing_alpha * probs + (1 - self.smoothing_alpha) * torch.mm(A, self.queue_probs)    
                
                mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs, softmax_x_ulb=False)

                # contrastive_x_ulb_w replace feats_x_ulb_w
                feats_w = torch.cat([contrastive_x_ulb_w, contrastive_x_lb],dim=0)
                probs_w = torch.cat([probs_orig, F.one_hot(y_lb, num_classes=self.num_classes)],dim=0)

                self.update_bank(feats_w, probs_w)

            # unsup_loss = self.consistency_loss(logits_x_ulb_s_0,
            #                               probs,
            #                               'ce',
            #                               mask=mask)
            unsup_loss = torch.zeros(1).cuda()

            # pseudo-label graph with self-loop
            Q = torch.mm(probs, probs.t())       
            Q.fill_diagonal_(1)    
            pos_mask = (Q >= self.contrast_p_cutoff).to(mask.dtype)
            Q = Q * pos_mask
            Q = Q / Q.sum(1, keepdim=True)

            contrast_loss = comatch_contrastive_loss(contrastive_x_ulb_s_0, contrastive_x_ulb_s_1, Q, T=self.T)

            contrastive_x_all = torch.cat([
                proto_proj,
                contrastive_x_lb,
                contrastive_x_ulb_s_0[maskbool],
                contrastive_x_ulb_s_1[maskbool],
                contrastive_x_ulb_s_0[~maskbool],
                contrastive_x_ulb_s_1[~maskbool]
            ], dim=0)

            # Concatenate labels and pseudo labels accordingly
            y_all = torch.cat([
                torch.arange(self.args.num_classes).cuda(),  # Prototype labels
                y_lb,  # Labeled data labels
                pseudo_label[maskbool],  # Pseudo labels for confident unlabeled data
                pseudo_label[maskbool],  # Duplicate for second set of confident unlabeled data
                (torch.arange((~maskbool).sum()) + self.args.num_classes).repeat(2).cuda()
                # Incremented labels for unconfident unlabeled data
            ], dim=0)

            # Initialize weights for all data points
            weights = torch.ones(y_all.shape[0]).cuda()
            P = (~maskbool).sum().item()  # Number of unconfident unlabeled data points

            # Apply different weights based on the data type
            weights[-2 * P:] *= self.args.lambda_ydown  # Apply down-weighting for unconfident unlabeled data
            weights[:self.args.num_classes] *= self.args.lambda_proto  # Apply weighting for prototypes

            # Apply lambda_yup for confident unlabeled data points
            proto_lb_len = proto_proj.size(0) + contrastive_x_lb.size(0)
            mask_true_len = maskbool.sum().item()
            start_index = proto_lb_len
            end_index = start_index + 2 * mask_true_len
            weights[start_index:end_index] *= self.args.lambda_yup

            supcon_loss = self.supcon_loss_weights(embeddings=contrastive_x_all, labels=y_all, weights=weights)
            total_loss = supcon_loss + self.lambda_c * contrast_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(supcon_loss=supcon_loss.item(),
                                         unsup_loss=unsup_loss.item(), 
                                         contrast_loss=contrast_loss.item(),
                                         total_loss=total_loss.item(),
                                         util_ratio=maskbool.float().mean().item(),
                                         pseudolabel_accuracy=((
                                                                       pseudo_label == y_ulb).float() * maskbool.float()).sum() / mask_sum.item() if mask_sum > 0 else 0)

        return out_dict, log_dict

    def get_save_dict(self):
        save_dict =  super().get_save_dict()
        save_dict['queue_feats'] = self.queue_feats.cpu()
        save_dict['queue_probs'] = self.queue_probs.cpu()
        save_dict['queue_size'] = self.queue_size
        save_dict['queue_ptr'] = self.queue_ptr
        save_dict['p_model'] = self.hooks_dict['DistAlignHook'].p_model.cpu() 
        save_dict['p_model_ptr'] = self.hooks_dict['DistAlignHook'].p_model_ptr.cpu()
        # save_dict['p_target'] = self.hooks_dict['DistAlignHook'].p_target.cpu() 
        # save_dict['p_target_ptr'] = self.hooks_dict['DistAlignHook'].p_target_ptr.cpu()
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.queue_feats = checkpoint['queue_feats'].cuda(self.gpu)
        self.queue_probs = checkpoint['queue_probs'].cuda(self.gpu)
        self.queue_size = checkpoint['queue_size']
        self.queue_ptr = checkpoint['queue_ptr']
        self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['DistAlignHook'].p_model_ptr = checkpoint['p_model_ptr'].cuda(self.args.gpu)
        # self.hooks_dict['DistAlignHook'].p_target = checkpoint['p_target'].cuda(self.args.gpu)
        # self.hooks_dict['DistAlignHook'].p_target_ptr = checkpoint['p_target_ptr'].cuda(self.args.gpu)
        return checkpoint

    def evaluate(self, eval_dest='eval', out_key='contrastive_feats', return_logits=False):
        """
        evaluation function overided from base class algorithmbase (the function is called in core/hook/evaluation)
        """
        self.model.eval()
        self.ema.apply_shadow()

        eval_loader = self.loader_dict[eval_dest]
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        # y_probs = []
        y_logits = []
        with torch.no_grad():
            for data in eval_loader:
                x = data['x_lb']
                y = data['y_lb']

                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)

                num_batch = y.shape[0]
                total_num += num_batch

                # similarity_to_proto = contrastive_x_ulb_w @ proto_proj.t()  # (N, K) = (N, D) @ (D, K) equivalent des softmax
                # print(f"similarity to proto first line : {similarity_to_proto[0, :]} and first label {y_ulb[0]}")
                #
                # pseudo_label = torch.argmax(similarity_to_proto, dim=1)
                # maskbool = torch.max(similarity_to_proto, dim=1)[0] > self.p_cutoff
                # mask_sum = maskbool.sum()  # number of samples with high confidence
                #
                # contrastive_x_all = torch.cat((contrastive_x_lb, contrastive_x_ulb_s_0[maskbool],
                #                                contrastive_x_ulb_s_1[maskbool], proto_proj), dim=0)

                out = self.model(x, contrastive=True)
                contrastive_feats = out[out_key]
                proto_proj = out['proto_proj']

                similarity_to_proto = contrastive_feats @ proto_proj.t()  # (N, K) = (N, D) @ (D, K) equivalent des softmax
                pred = torch.argmax(similarity_to_proto, dim=1)


                # loss = F.cross_entropy(logits, y, reduction='mean', ignore_index=-1)
                y_true.extend(y.cpu().tolist())
                y_pred.extend(pred.cpu().tolist())
                # y_logits.append(logits.cpu().numpy())
                # total_loss += loss.item() * num_batch
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        # y_logits = np.concatenate(y_logits)
        top1 = accuracy_score(y_true, y_pred)
        balanced_top1 = balanced_accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        F1 = f1_score(y_true, y_pred, average='macro')

        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        self.ema.restore()
        self.model.train()

        eval_dict = {eval_dest + '/top-1-acc': top1,  # eval_dest + '/loss': total_loss / total_num,
                     eval_dest + '/balanced_acc': balanced_top1, eval_dest + '/precision': precision,
                     eval_dest + '/recall': recall, eval_dest + '/F1': F1}
        # if return_logits:
        #     eval_dict[eval_dest + '/logits'] = y_logits
        return eval_dict
    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--contrast_p_cutoff', float, 0.8),
            SSL_Argument('--contrast_loss_ratio', float, 1.0),
            SSL_Argument('--proj_size', int, 128),
            SSL_Argument('--queue_batch', int, 128),
            SSL_Argument('--smoothing_alpha', float, 0.9),
            SSL_Argument('--da_len', int, 256),
        ]
