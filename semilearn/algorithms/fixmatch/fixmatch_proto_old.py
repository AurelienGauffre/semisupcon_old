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

            if self.args.loss == "simclr":
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
