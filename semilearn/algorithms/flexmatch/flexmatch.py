# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from .utils import FlexMatchThresholdingHook

from semilearn.core import AlgorithmBase
from semilearn.core.algorithmbase import SupConLossWeights
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool

from pytorch_metric_learning import losses

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix
import numpy as np

@ALGORITHMS.register('flexmatch')
class FlexMatch(AlgorithmBase):
    """
        FlexMatch algorithm (https://arxiv.org/abs/2110.08263).

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
            - ulb_dest_len (`int`):
                Length of unlabeled data
            - thresh_warmup (`bool`, *optional*, default to `True`):
                If True, warmup the confidence threshold, so that at the beginning of the training, all estimated
                learning effects gradually rise from 0 until the number of unused unlabeled data is no longer
                predominant

        """

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        # flexmatch specified arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label, thresh_warmup=args.thresh_warmup)

    def init(self, T, p_cutoff, hard_label=True, thresh_warmup=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self.thresh_warmup = thresh_warmup

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FlexMatchThresholdingHook(ulb_dest_len=self.args.ulb_dest_len, num_classes=self.num_classes,
                                                     thresh_warmup=self.args.thresh_warmup), "MaskingHook")
        #self.register_hook(FixedThresholdingHook(), "MaskingHook") # celui de Fixmatch
        super().set_hooks()

    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s, y_ulb):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb)
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb_w, 'x_ulb_s': feats_x_ulb_s}

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())

            # if distribution alignment hook is registered, call it 
            # this is implemented for imbalanced algorithm - CReST
            if self.registered_hook("DistAlignHook"):
                probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())

            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False,
                                  idx_ulb=idx_ulb)
            mask_sum = mask.bool().sum()


            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)

            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                               pseudo_label,
                                               'ce',
                                               mask=mask)

            total_loss = sup_loss + self.lambda_u * unsup_loss





        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                         unsup_loss=unsup_loss.item(),
                                         total_loss=total_loss.item(),
                                         util_ratio=mask.float().mean().item(),
                                         pseudolabel_accuracy=((torch.argmax(logits_x_ulb_w,
                                                                             dim=1) == y_ulb).float() * mask).sum() / mask_sum.item() if mask_sum > 0 else 0
                                         )

        return out_dict, log_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['classwise_acc'] = self.hooks_dict['MaskingHook'].classwise_acc.cpu()
        save_dict['selected_label'] = self.hooks_dict['MaskingHook'].selected_label.cpu()
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['MaskingHook'].classwise_acc = checkpoint['classwise_acc'].cuda(self.gpu)
        self.hooks_dict['MaskingHook'].selected_label = checkpoint['selected_label'].cuda(self.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--thresh_warmup', str2bool, True),
        ]


@ALGORITHMS.register('flexmatch_contrastive')
class FlexMatchContrastive(AlgorithmBase):
    """
        FlexMatch algorithm (https://arxiv.org/abs/2110.08263).

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
            - ulb_dest_len (`int`):
                Length of unlabeled data
            - thresh_warmup (`bool`, *optional*, default to `True`):
                If True, warmup the confidence threshold, so that at the beginning of the training, all estimated
                learning effects gradually rise from 0 until the number of unused unlabeled data is no longer
                predominant

        """

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        # flexmatch specified arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label, thresh_warmup=args.thresh_warmup)
        self.supcon_loss = losses.SupConLoss()
        self.supcon_loss_weights = SupConLossWeights()
        self.is_contrastive = True

    def init(self, T, p_cutoff, hard_label=True, thresh_warmup=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self.thresh_warmup = thresh_warmup

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FlexMatchThresholdingHook(ulb_dest_len=self.args.ulb_dest_len, num_classes=self.num_classes,
                                                     thresh_warmup=self.args.thresh_warmup), "MaskingHook")
        super().set_hooks()

    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s_0, x_ulb_s_1, y_ulb):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                if self.use_cat:  # does not support detach of CE
                    inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s_0, x_ulb_s_1))
                    outputs = self.model(inputs, contrastive=self.is_contrastive)
                    contrastive_x = outputs['contrastive_feats']
                    contrastive_x_lb = contrastive_x[:num_lb]
                    contrastive_x_ulb_w, contrastive_x_ulb_s_0, contrastive_x_ulb_s_1 = contrastive_x[num_lb:].chunk(3)
                    proto_proj = outputs['proto_proj']
            else:
                raise ValueError("SemiSupConProto does not support non-cat mode currently")

            feat_dict = {'x_lb': contrastive_x_lb, 'x_ulb_w': contrastive_x_ulb_w,
                         'x_ulb_s': [contrastive_x_ulb_s_0, contrastive_x_ulb_s_1]}

            similarity_to_proto = contrastive_x_ulb_w @ proto_proj.t()
            pseudo_label = torch.argmax(similarity_to_proto, dim=1)

            if self.args.pl == "softmax":
                similarity_to_proto = torch.softmax((similarity_to_proto + 1) / 2 / self.args.pl_temp, dim=1)
            # print(
            #     f"before soft{similarity_to_proto.max(dim=1)[0].max(dim=0)[0].item()} {similarity_to_proto.max(dim=1)[0].mean(dim=0).item()}")
            prob = torch.softmax((similarity_to_proto + 1) / 2 / self.args.pl_temp, dim=1)

            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=similarity_to_proto, softmax_x_ulb=False,
                                  idx_ulb=idx_ulb)
            # convert binarty float mask of 0 and 1 to boolean mask
            mask = mask.bool()

            mask_sum = mask.sum()

            if self.registered_hook("DistAlignHook"):
                prob = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=prob.detach())

            pseudo_label = torch.argmax(similarity_to_proto, dim=1)

            # Your favorite loss here, E7 here (onlysupcon avec meme poids partout)
            contrastive_x_all = torch.cat(
                (proto_proj, contrastive_x_lb, contrastive_x_ulb_s_0[mask], contrastive_x_ulb_s_1[mask],
                 contrastive_x_ulb_s_0[~mask], contrastive_x_ulb_s_1[~mask]),
                dim=0)
            y_all = torch.cat(
                (torch.arange(self.args.num_classes).cuda(), y_lb, pseudo_label[mask], pseudo_label[mask],
                 (torch.arange(sum(~mask)).cuda() + self.args.num_classes).repeat(2)),
                dim=0)

            P = sum(~mask)
            weights = torch.ones(y_all.shape[0]).cuda()
            weights[-2 * P:] *= self.args.lambda_proto
            supcon_loss = self.supcon_loss_weights(embeddings=contrastive_x_all, labels=y_all,
                                                   weights=weights)
            unsup_loss = torch.zeros(1).cuda()
            total_loss = supcon_loss

            #
            #
            # unsup_loss = torch.zeros(1).cuda()
            # supcon_loss = self.supcon_loss(embeddings=contrastive_x_all, labels=y_all)




            total_loss = supcon_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(supcon_loss=supcon_loss.item(),
                                         unsup_loss=unsup_loss.item(),
                                         total_loss=total_loss.item(),
                                         util_ratio=mask.float().mean().item(),
                                         pseudolabel_accuracy=((
                                                                       pseudo_label == y_ulb).float() * mask.float()).sum() / mask_sum.item() if mask_sum > 0 else 0)

        return out_dict, log_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['classwise_acc'] = self.hooks_dict['MaskingHook'].classwise_acc.cpu()
        save_dict['selected_label'] = self.hooks_dict['MaskingHook'].selected_label.cpu()
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['MaskingHook'].classwise_acc = checkpoint['classwise_acc'].cuda(self.gpu)
        self.hooks_dict['MaskingHook'].selected_label = checkpoint['selected_label'].cuda(self.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--thresh_warmup', str2bool, True),
        ]

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