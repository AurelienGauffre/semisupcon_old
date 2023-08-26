# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from semilearn.core.algorithmbase import AlgorithmBase, SupConLossWeights
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix

from pytorch_metric_learning import losses


@ALGORITHMS.register('fixmatch')
class FixMatch(AlgorithmBase):
    """
        FixMatch algorithm (https://arxiv.org/abs/2001.07685).

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
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        # fixmatch specified arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label)

    def init(self, T, p_cutoff, hard_label=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s, y_ulb):
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
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)
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
                                         # else float('nan')
                                         )
        return out_dict, log_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
        ]


@ALGORITHMS.register('semisupcon')
class SemiSupCon(AlgorithmBase):
    """
        FixMatch algorithm (https://arxiv.org/abs/2001.07685).

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
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        # fixmatch specified arguments

        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label)
        self.supcon_loss = losses.SupConLoss()

    def init(self, T, p_cutoff, hard_label=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s_0, x_ulb_s_1, y_ulb):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:  # does not support detach of CE
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s_0, x_ulb_s_1))
                outputs = self.model(inputs, contrastive=True)
                logits, contrastive_x = outputs['logits'], outputs['contrastive_feats']
                logits_x_lb, contrastive_x_lb = logits[:num_lb], contrastive_x[:num_lb]
                logits_x_ulb_w, logits_x_ulb_s_0, logits_x_ulb_s_1 = logits[num_lb:].chunk(3)
                contrastive_x_ulb_w, contrastive_x_ulb_s_0, contrastive_x_ulb_s_1 = contrastive_x[num_lb:].chunk(3)

            else:
                DETACH = True
                outs_x_lb = self.model(x_lb, contrastive=True, detach_ce=DETACH)
                logits_x_lb, contrastive_x_lb = outs_x_lb['logits'], outs_x_lb['contrastive_feats']
                outs_x_ulb_s_0 = self.model(x_ulb_s_0, contrastive=True, detach_ce=DETACH)
                logits_x_ulb_s_0, contrastive_x_ulb_s_0 = outs_x_ulb_s_0['logits'], outs_x_ulb_s_0['contrastive_feats']
                outs_x_ulb_s_1 = self.model(x_ulb_s_1, contrastive=True, detach_ce=DETACH)
                logits_x_ulb_s_1, contrastive_x_ulb_s_1 = outs_x_ulb_s_1['logits'], outs_x_ulb_s_1['contrastive_feats']

                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w, contrastive=True)
                    logits_x_ulb_w, contrastive_x_ulb_w = outs_x_ulb_w['logits'], outs_x_ulb_w['contrastive_feats']

            # feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb_w, 'x_ulb_s': [feats_x_ulb_s_0, feats_x_ulb_s_1]}
            feat_dict = {'x_lb': contrastive_x_lb, 'x_ulb_w': contrastive_x_ulb_w,
                         'x_ulb_s': [contrastive_x_ulb_s_0, contrastive_x_ulb_s_1]}

            # Computation of mask/pseudo labels
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())  # torch.softmax(logits, dim=-1)
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)
            maskbool = mask.bool()
            mask_sum = maskbool.sum()
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)

            contrastive_x_all = torch.cat(
                (contrastive_x_lb, contrastive_x_ulb_s_0[maskbool], contrastive_x_ulb_s_1[maskbool]), dim=0)
            y_all = torch.cat((y_lb, pseudo_label[maskbool], pseudo_label[maskbool]), dim=0)

            if self.args.loss == "only_unsup":
                simclr_loss = self.supcon_loss(
                    embeddings=torch.cat((contrastive_x_ulb_s_0, contrastive_x_ulb_s_1)),
                    labels=torch.arange(len(maskbool)).repeat(2))
                total_loss = simclr_loss
                ce_loss_sup = torch.zeros(1)
                ce_loss_unsup = torch.zeros(1)
                ce_loss = ce_loss_sup + ce_loss_unsup
                supcon_loss = torch.zeros(1)
            elif self.args.loss == "all_withoutsimclr":
                ce_loss_sup = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
                # ce_loss_unsup = self.ce_loss(logits_x_ulb_w[maskbool], pseudo_label[maskbool], reduction='mean')
                ce_loss_unsup = self.consistency_loss(logits_x_ulb_s_0,
                                                      pseudo_label,
                                                      'ce',
                                                      mask=mask) + self.consistency_loss(logits_x_ulb_s_1,
                                                                                         pseudo_label,
                                                                                         'ce',
                                                                                         mask=mask)
                # BIG CHANGE : the ce_loss_unsuper is removed
                ce_loss = ce_loss_sup + ce_loss_unsup
                supcon_loss = self.supcon_loss(embeddings=contrastive_x_all, labels=y_all)
                # simclr_loss_light = self.supcon_loss(
                #     embeddings=torch.cat((contrastive_x_ulb_s_0[~maskbool], contrastive_x_ulb_s_1[~maskbool])),
                #                          labels=torch.arange(sum(~maskbool)).repeat(2))

                # simclr_loss_heavy = self.supcon_loss(
                #     embeddings=torch.cat((contrastive_x_ulb_s_0[~maskbool], contrastive_x_ulb_s_1[~maskbool])),
                #                          labels=torch.arange(sum(~maskbool)).repeat(2))

                total_loss = supcon_loss + self.lambda_u * ce_loss  # + 0.5*simclr_loss
            elif self.args.loss == "all_withoutsimclr_unsupcedetached":
                ce_loss_sup = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
                # ce_loss_unsup = self.ce_loss(logits_x_ulb_w[maskbool], pseudo_label[maskbool], reduction='mean')
                ce_loss_unsup = self.consistency_loss(logits_x_ulb_s_0.detach(),
                                                      pseudo_label,
                                                      'ce',
                                                      mask=mask) + self.consistency_loss(logits_x_ulb_s_1.detach(),
                                                                                         pseudo_label,
                                                                                         'ce',
                                                                                         mask=mask)
                # BIG CHANGE : the ce_loss_unsuper is removed
                ce_loss = ce_loss_sup + ce_loss_unsup
                supcon_loss = self.supcon_loss(embeddings=contrastive_x_all, labels=y_all)
                # simclr_loss_light = self.supcon_loss(
                #     embeddings=torch.cat((contrastive_x_ulb_s_0[~maskbool], contrastive_x_ulb_s_1[~maskbool])),
                #                          labels=torch.arange(sum(~maskbool)).repeat(2))

                # simclr_loss_heavy = self.supcon_loss(
                #     embeddings=torch.cat((contrastive_x_ulb_s_0[~maskbool], contrastive_x_ulb_s_1[~maskbool])),
                #                          labels=torch.arange(sum(~maskbool)).repeat(2))

                total_loss = supcon_loss + self.lambda_u * ce_loss  # + 0.5*simclr_loss
            elif self.args.loss == "all_withoutsimclr_withoutunsupce":
                ce_loss_sup = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
                # ce_loss_unsup = self.ce_loss(logits_x_ulb_w[maskbool], pseudo_label[maskbool], reduction='mean')

                # BIG CHANGE : the ce_loss_unsuper is removed
                ce_loss = ce_loss_sup
                ce_loss_unsup = torch.tensor(float('nan'))
                supcon_loss = self.supcon_loss(embeddings=contrastive_x_all, labels=y_all)
                # simclr_loss_light = self.supcon_loss(
                #     embeddings=torch.cat((contrastive_x_ulb_s_0[~maskbool], contrastive_x_ulb_s_1[~maskbool])),
                #                          labels=torch.arange(sum(~maskbool)).repeat(2))

                # simclr_loss_heavy = self.supcon_loss(
                #     embeddings=torch.cat((contrastive_x_ulb_s_0[~maskbool], contrastive_x_ulb_s_1[~maskbool])),
                #                          labels=torch.arange(sum(~maskbool)).repeat(2))

                total_loss = supcon_loss + self.lambda_u * ce_loss  # + 0.5*simclr_loss

            elif self.args.loss == "ponderate_ce_loss_unsup":
                fraction = torch.mean(mask)
                ce_loss_sup = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
                ce_loss_unsup = self.consistency_loss(logits_x_ulb_s_0,
                                                      pseudo_label,
                                                      'ce',
                                                      mask=mask) + self.consistency_loss(logits_x_ulb_s_1,
                                                                                         pseudo_label,
                                                                                         'ce',
                                                                                         mask=mask)

                ce_loss = ce_loss_sup + ce_loss_unsup * fraction
                supcon_loss = self.supcon_loss(embeddings=contrastive_x_all, labels=y_all)
                total_loss = supcon_loss + self.lambda_u * ce_loss

            elif self.args.loss == "ponderate_simclr_loss_light":
                fraction_pseudo_labeled = torch.mean(mask)
                ce_loss_sup = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
                ce_loss_unsup = self.consistency_loss(logits_x_ulb_s_0,
                                                      pseudo_label,
                                                      'ce',
                                                      mask=mask) + self.consistency_loss(logits_x_ulb_s_1,
                                                                                         pseudo_label,
                                                                                         'ce',
                                                                                         mask=mask)

                ce_loss = ce_loss_sup + ce_loss_unsup
                supcon_loss = self.supcon_loss(embeddings=contrastive_x_all, labels=y_all)
                simclr_loss_light = self.supcon_loss(
                    embeddings=torch.cat((contrastive_x_ulb_s_0[~maskbool], contrastive_x_ulb_s_1[~maskbool])),
                    labels=torch.arange(sum(~maskbool)).repeat(2)) * (1 - fraction_pseudo_labeled)
                total_loss = supcon_loss + self.lambda_u * ce_loss + simclr_loss_light
            elif self.args.loss == "ponderate_simclr_loss_heavy":
                fraction_pseudo_labeled = torch.mean(mask)
                ce_loss_sup = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
                ce_loss_unsup = self.consistency_loss(logits_x_ulb_s_0,
                                                      pseudo_label,
                                                      'ce',
                                                      mask=mask) + self.consistency_loss(logits_x_ulb_s_1,
                                                                                         pseudo_label,
                                                                                         'ce',
                                                                                         mask=mask)

                ce_loss = ce_loss_sup + ce_loss_unsup
                supcon_loss = self.supcon_loss(embeddings=contrastive_x_all, labels=y_all)
                simclr_loss_light = self.supcon_loss(
                    embeddings=torch.cat((contrastive_x_ulb_s_0, contrastive_x_ulb_s_1)),
                    labels=torch.arange(len(mask)).repeat(2)) * (1 - fraction_pseudo_labeled)
                total_loss = supcon_loss + self.lambda_u * ce_loss + simclr_loss_light

            elif self.args.loss == "simclr_loss_heavy":
                fraction_pseudo_labeled = torch.mean(mask)
                ce_loss_sup = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
                ce_loss_unsup = self.consistency_loss(logits_x_ulb_s_0,
                                                      pseudo_label,
                                                      'ce',
                                                      mask=mask) + self.consistency_loss(logits_x_ulb_s_1,
                                                                                         pseudo_label,
                                                                                         'ce',
                                                                                         mask=mask)

                ce_loss = ce_loss_sup + ce_loss_unsup
                supcon_loss = self.supcon_loss(embeddings=contrastive_x_all, labels=y_all)
                simclr_loss_light = self.supcon_loss(
                    embeddings=torch.cat((contrastive_x_ulb_s_0, contrastive_x_ulb_s_1)),
                    labels=torch.arange(len(mask)).repeat(2))
                total_loss = supcon_loss + self.lambda_u * ce_loss + simclr_loss_light

            elif self.args.loss == "knn":  # pondereate loss unsup
                fraction = torch.mean(mask)
                ce_loss_sup = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
                ce_loss_unsup = self.consistency_loss(logits_x_ulb_s_0, pseudo_label, 'ce', mask=mask) \
                                + self.consistency_loss(logits_x_ulb_s_1, pseudo_label, 'ce', mask=mask)
                ce_loss = ce_loss_sup + ce_loss_unsup * fraction
                supcon_loss = self.supcon_loss(embeddings=contrastive_x_all, labels=y_all)
                total_loss = supcon_loss + self.lambda_u * ce_loss

            out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
            log_dict = self.process_log_dict(ce_loss=ce_loss.item(),
                                             ce_loss_sup=ce_loss_sup.item(),
                                             ce_loss_unsup=ce_loss_unsup.item(),
                                             supcon_loss=supcon_loss.item(),
                                             total_loss=total_loss.item(),
                                             util_ratio=mask.float().mean().item(),
                                             pseudolabel_accuracy=((torch.argmax(logits_x_ulb_w,
                                                                                 dim=1) == y_ulb).float() * mask).sum() / mask_sum.item() if mask_sum > 0 else 0
                                             # else float('nan')

                                             )
            # pseulabel_accuracy=)

            return out_dict, log_dict


@ALGORITHMS.register('semisupconproto')
class SemiSupConProto(AlgorithmBase):
    """


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
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        # fixmatch specified arguments

        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label)
        self.supcon_loss = losses.SupConLoss()
        self.supcon_loss_weights = SupConLossWeights()

    def init(self, T, p_cutoff, hard_label=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s_0, x_ulb_s_1, y_ulb):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:  # does not support detach of CE
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s_0, x_ulb_s_1))
                outputs = self.model(inputs, contrastive=True)
                contrastive_x = outputs['contrastive_feats']
                contrastive_x_lb = contrastive_x[:num_lb]
                contrastive_x_ulb_w, contrastive_x_ulb_s_0, contrastive_x_ulb_s_1 = contrastive_x[num_lb:].chunk(3)
                proto_proj = outputs['proto_proj']
            else:
                raise ValueError("SemiSupConProto does not support non-cat mode currently")

            feat_dict = {'x_lb': contrastive_x_lb, 'x_ulb_w': contrastive_x_ulb_w,
                         'x_ulb_s': [contrastive_x_ulb_s_0, contrastive_x_ulb_s_1]}

            similarity_to_proto = contrastive_x_ulb_w @ proto_proj.t()  # (N, K) = (N, D) @ (D, K) equivalent des softmax
            print(f"similarity to proto first line : {similarity_to_proto[0, :]} and first label {y_ulb[0]}")

            pseudo_label = torch.argmax(similarity_to_proto, dim=1)
            maskbool = torch.max(similarity_to_proto, dim=1)[0] > self.p_cutoff
            mask_sum = maskbool.sum()  # number of samples with high confidence

            contrastive_x_all = torch.cat(
                (contrastive_x_lb, contrastive_x_ulb_s_0[maskbool], contrastive_x_ulb_s_1[maskbool], proto_proj), dim=0)
            y_all = torch.cat(
                (y_lb, pseudo_label[maskbool], pseudo_label[maskbool], torch.arange(self.args.num_classes).cuda()),
                dim=0)  # TODO Ne pas hardcoder le nombre de classes

            if self.args.loss == "full_supcon":
                contrastive_x_all = torch.cat(
                    (contrastive_x_all, contrastive_x_ulb_s_0[~maskbool], contrastive_x_ulb_s_1[~maskbool]), dim=0)
                y_all = torch.cat((y_all, (torch.arange(sum(~maskbool)).cuda() + self.args.num_classes).repeat(2)),
                                  dim=0)  # TODO Ne pas hardcoder le nombre de classes

                supcon_loss = self.supcon_loss(embeddings=contrastive_x_all, labels=y_all)

                total_loss = supcon_loss
            elif self.args.loss == "full_supcon_weights":
                contrastive_x_all = torch.cat(
                    (contrastive_x_all, contrastive_x_ulb_s_0[~maskbool], contrastive_x_ulb_s_1[~maskbool]), dim=0)
                y_all = torch.cat((y_all, (torch.arange(sum(~maskbool)).cuda() + self.args.num_classes).repeat(2)),
                                  dim=0)  # TODO Ne pas hardcoder le nombre de classes

                supcon_loss = self.supcon_loss_weights(embeddings=contrastive_x_all, labels=y_all,weighs=torch.ones(y_all.shape[0]))

                total_loss = supcon_loss

            elif self.args.loss == "supcon_simclr(remaining)":
                "Supcon to labeled and pseudolabels + simclr only on non pseudo labeled examples"

                supcon_loss = self.supcon_loss(embeddings=contrastive_x_all, labels=y_all)
                simclr = self.supcon_loss(
                    embeddings=torch.cat((contrastive_x_ulb_s_0[~maskbool], contrastive_x_ulb_s_1[~maskbool])),
                    labels=torch.arange(sum(~maskbool)).repeat(2))
                total_loss = supcon_loss + simclr

            elif self.args.loss == "supcon_simclr(all)":
                "Supcon to labeled and pseudolabels + simclr on all unsupervised labels"
                supcon_loss = self.supcon_loss(embeddings=contrastive_x_all, labels=y_all)
                simclr = self.supcon_loss(
                    embeddings=torch.cat((contrastive_x_ulb_s_0, contrastive_x_ulb_s_1)),
                    labels=torch.arange(maskbool.shape[0]).repeat(2))
                total_loss = supcon_loss + simclr

            else:
                raise ValueError("Unknown loss type")

            out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
            log_dict = self.process_log_dict(supcon_loss=supcon_loss.item(),
                                             total_loss=total_loss.item(),
                                             util_ratio=maskbool.float().mean().item(),
                                             pseudolabel_accuracy=((
                                                                               pseudo_label == y_ulb).float() * maskbool.float()).sum() / mask_sum.item() if mask_sum > 0 else 0)

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

                out = self.model(x)
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
