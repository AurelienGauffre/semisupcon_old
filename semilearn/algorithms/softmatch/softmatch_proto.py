# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from .utils import SoftMatchWeightingHook
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core import SupConLossWeights
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, DistAlignEMAHook
from semilearn.algorithms.utils import SSL_Argument, str2bool

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix

@ALGORITHMS.register('softmatch_proto')
class SoftMatchProto(AlgorithmBase):
    """
        SoftMatch algorithm (https://openreview.net/forum?id=ymt1zQXBDiF&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2023%2FConference%2FAuthors%23your-submissions)).

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
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
            - ema_p (`float`):
                exponential moving average of probability update
        """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        self.init(T=args.T, hard_label=args.hard_label, dist_align=args.dist_align, dist_uniform=args.dist_uniform, ema_p=args.ema_p, n_sigma=args.n_sigma, per_class=args.per_class)
    
    def init(self, T, hard_label=True, dist_align=True, dist_uniform=True, ema_p=0.999, n_sigma=2, per_class=False):
        self.T = T
        self.use_hard_label = hard_label
        self.dist_align = dist_align
        self.dist_uniform = dist_uniform
        self.ema_p = ema_p
        self.n_sigma = n_sigma
        self.per_class = per_class
        self.supcon_loss_weights = SupConLossWeights()
    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(
            DistAlignEMAHook(num_classes=self.num_classes, momentum=self.args.ema_p, p_target_type='uniform' if self.args.dist_uniform else 'model'), 
            "DistAlignHook")
        self.register_hook(SoftMatchWeightingHook(num_classes=self.num_classes, n_sigma=self.args.n_sigma, momentum=self.args.ema_p, per_class=self.args.per_class), "MaskingHook")
        super().set_hooks()    

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s_0, x_ulb_s_1):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s_0, x_ulb_s_1))
                outputs = self.model(inputs, contrastive=True)
                contrastive_x = outputs['contrastive_feats']
                contrastive_x_lb = contrastive_x[:num_lb]
                contrastive_x_ulb_w, contrastive_x_ulb_s_0, contrastive_x_ulb_s_1 = contrastive_x[num_lb:].chunk(3)
                proto_proj = outputs['proto_proj']

            else:
               raise NotImplementedError("Not implemented for contrastive models")


            #feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}
            feat_dict = {'x_lb': contrastive_x_lb, 'x_ulb_w': contrastive_x_ulb_w,
                         'x_ulb_s': [contrastive_x_ulb_s_0, contrastive_x_ulb_s_1]}

            similarity_to_proto = contrastive_x_ulb_w @ proto_proj.t()  # (N, K) = (N, D) @ (D, K) equivalent des softmax
            pseudo_label = torch.argmax(similarity_to_proto, dim=1)
            # sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            #probs_x_lb = torch.softmax(logits_x_lb.detach(), dim=-1)
            #probs_x_ulb_w = torch.softmax(logits_x_ulb_w.detach(), dim=-1)


            probs_x_lb = torch.softmax((contrastive_x_lb @ proto_proj.t() + 1) / 2 / self.args.pl_temp, dim=1) #utilisé just pour le dist align
            probs_x_ulb_w= torch.softmax((similarity_to_proto + 1) / 2 / self.args.pl_temp, dim=1)
            similarity_to_proto = torch.softmax((similarity_to_proto + 1) / 2 / self.args.pl_temp, dim=1)

            # uniform distribution alignment 
            probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w, probs_x_lb=probs_x_lb)

            # calculate weight
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

            # generate unlabeled targets using pseudo label hook

            # calculate loss

            contrastive_x_all = torch.cat([
                proto_proj,
                contrastive_x_lb,
                contrastive_x_ulb_s_0,
                contrastive_x_ulb_s_1,
                contrastive_x_ulb_s_0,
                contrastive_x_ulb_s_1,
            ], dim=0)

            # Concatenate labels and pseudo labels accordingly
            y_all = torch.cat([
                torch.arange(self.args.num_classes).cuda(),  # Prototype labels
                y_lb,  # Labeled data labels
                pseudo_label,  # Pseudo labels for confident unlabeled data
                pseudo_label,  # Duplicate for second set of confident unlabeled data
                (torch.arange(contrastive_x_ulb_s_0.shape[0]) + self.args.num_classes).repeat(2).cuda()
                # Incremented labels for unconfident unlabeled data
            ], dim=0)

            # Initialize weights for all data points
            weights = torch.ones(y_all.shape[0]).cuda()

            n_u = pseudo_label.shape[0]
            # Apply different weights based on the data type
            weights[-2 * n_u:] *= (self.args.lambda_ydown * (1 - mask)).repeat(2)  # Apply down-weighting for unconfident unlabeled data
            weights[:self.args.num_classes] *= self.args.lambda_proto  # Apply weighting for prototypes

            # Apply lambda_yup for confident unlabeled data points
            proto_lb_len = proto_proj.size(0) + contrastive_x_lb.size(0)
            start_index = proto_lb_len
            end_index = start_index + 2 * n_u
            weights[start_index:end_index] *= self.args.lambda_yup * mask.repeat(2)

            # Compute the supervised contrastive loss
            supcon_loss = self.supcon_loss_weights(embeddings=contrastive_x_all, labels=y_all, weights=weights)

            # Assuming unsupervised loss is not used in this context
            unsup_loss = torch.zeros(1).cuda()  # Placeholder for unsupervised loss, if necessary

            # Final total loss is just the supervised contrastive loss in this case
            total_loss = supcon_loss


        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(supcon_loss=supcon_loss.item(),
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict
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

                out = self.model(x, contrastive=self.is_contrastive)
                contrastive_feats = out[out_key]
                proto_proj = out['proto_proj']
                if self.is_contrastive:
                    similarity_to_proto = contrastive_feats @ proto_proj.t()  # (N, K) = (N, D) @ (D, K) equivalent des softmax
                    pred = torch.argmax(similarity_to_proto, dim=1)
                else:
                    pred = torch.argmax(contrastive_feats, dim=1)

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

    # TODO: change these
    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['p_model'] = self.hooks_dict['DistAlignHook'].p_model.cpu()
        save_dict['p_target'] = self.hooks_dict['DistAlignHook'].p_target.cpu()
        save_dict['prob_max_mu_t'] = self.hooks_dict['MaskingHook'].prob_max_mu_t.cpu()
        save_dict['prob_max_var_t'] = self.hooks_dict['MaskingHook'].prob_max_var_t.cpu()
        return save_dict


    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['DistAlignHook'].p_target = checkpoint['p_target'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].prob_max_mu_t = checkpoint['prob_max_mu_t'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].prob_max_var_t = checkpoint['prob_max_var_t'].cuda(self.args.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--dist_align', str2bool, True),
            SSL_Argument('--dist_uniform', str2bool, True),
            SSL_Argument('--ema_p', float, 0.999),
            SSL_Argument('--n_sigma', int, 2),
            SSL_Argument('--per_class', str2bool, False),
        ]
