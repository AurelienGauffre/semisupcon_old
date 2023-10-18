import torch
from semilearn.core.algorithmbase import AlgorithmBase, SupConLossWeights
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.core.hooks import Hook, get_priority, CheckpointHook, TimerHook, LoggingHook, DistSamplerSeedHook, \
    ParamUpdateHook, EvaluationHook, EMAHook, WANDBHook, AimHook
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix

from pytorch_metric_learning import losses


@ALGORITHMS.register('unsup')
class Unsup(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
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
        self.register_hook(ParamUpdateHook(), None, "HIGHEST")
        self.register_hook(EMAHook(), None, "HIGH")
        self.register_hook(CheckpointHook(), None, "HIGH")
        self.register_hook(DistSamplerSeedHook(), None, "NORMAL")
        self.register_hook(TimerHook(), None, "LOW")
        self.register_hook(LoggingHook(), None, "LOWEST")
        if self.args.use_wandb:
            self.register_hook(WANDBHook(), None, "LOWEST")
        if self.args.use_aim:
            self.register_hook(AimHook(), None, "LOWEST")
    def train_step(self, x_ulb_s_0, x_ulb_s_1):

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:  # does not support detach of CE
                inputs = torch.cat((x_ulb_s_0, x_ulb_s_1))
                outputs = self.model(inputs)
                contrastive_x = outputs['contrastive_feats']
                contrastive_x_ulb_s_0, contrastive_x_ulb_s_1 = contrastive_x.chunk(2)

            else:
                raise ValueError("unsup does not support non-cat mode currently")

            feat_dict = {'x_ulb_s': [contrastive_x_ulb_s_0, contrastive_x_ulb_s_1]}

            if self.args.loss == "simclr":
                simclr = self.supcon_loss(
                    embeddings=torch.cat((contrastive_x_ulb_s_0, contrastive_x_ulb_s_1)),
                    labels=torch.arange(len(contrastive_x_ulb_s_0)).repeat(2))

                total_loss = simclr


            else:
                raise ValueError("Unknown loss type")

            out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
            log_dict = self.process_log_dict(
                                             total_loss=total_loss.item(),
                                             )


            return out_dict, log_dict

    def evaluate(self, eval_dest='eval', out_key='contrastive_feats', return_logits=False):
        """
        evaluation function overided from base class algorithmbase (the function is called in core/hook/evaluation)
        """
        # self.model.eval()
        # self.ema.apply_shadow()
        #
        # eval_loader = self.loader_dict[eval_dest]
        # total_loss = 0.0
        # total_num = 0.0
        # y_true = []
        # y_pred = []
        # # y_probs = []
        # y_logits = []
        # with torch.no_grad():
        #     for data in eval_loader:
        #         x = data['x_lb']
        #         y = data['y_lb']
        #
        #         if isinstance(x, dict):
        #             x = {k: v.cuda(self.gpu) for k, v in x.items()}
        #         else:
        #             x = x.cuda(self.gpu)
        #         y = y.cuda(self.gpu)
        #
        #         num_batch = y.shape[0]
        #         total_num += num_batch
        #
        #         # similarity_to_proto = contrastive_x_ulb_w @ proto_proj.t()  # (N, K) = (N, D) @ (D, K) equivalent des softmax
        #         # print(f"similarity to proto first line : {similarity_to_proto[0, :]} and first label {y_ulb[0]}")
        #         #
        #         # pseudo_label = torch.argmax(similarity_to_proto, dim=1)
        #         # maskbool = torch.max(similarity_to_proto, dim=1)[0] > self.p_cutoff
        #         # mask_sum = maskbool.sum()  # number of samples with high confidence
        #         #
        #         # contrastive_x_all = torch.cat((contrastive_x_lb, contrastive_x_ulb_s_0[maskbool],
        #         #                                contrastive_x_ulb_s_1[maskbool], proto_proj), dim=0)
        #
        #         out = self.model(x)
        #         contrastive_feats = out[out_key]
        #         proto_proj = out['proto_proj']
        #         similarity_to_proto = contrastive_feats @ proto_proj.t()  # (N, K) = (N, D) @ (D, K) equivalent des softmax
        #         pred = torch.argmax(similarity_to_proto, dim=1)
        #
        #         # loss = F.cross_entropy(logits, y, reduction='mean', ignore_index=-1)
        #         y_true.extend(y.cpu().tolist())
        #         y_pred.extend(pred.cpu().tolist())
        #         # y_logits.append(logits.cpu().numpy())
        #         # total_loss += loss.item() * num_batch
        # y_true = np.array(y_true)
        # y_pred = np.array(y_pred)
        # # y_logits = np.concatenate(y_logits)
        # top1 = accuracy_score(y_true, y_pred)
        # balanced_top1 = balanced_accuracy_score(y_true, y_pred)
        # precision = precision_score(y_true, y_pred, average='macro')
        # recall = recall_score(y_true, y_pred, average='macro')
        # F1 = f1_score(y_true, y_pred, average='macro')
        #
        # cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        # self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        # self.ema.restore()
        # self.model.train()

        # eval_dict = {eval_dest + '/top-1-acc': top1,  # eval_dest + '/loss': total_loss / total_num,
        #              eval_dest + '/balanced_acc': balanced_top1, eval_dest + '/precision': precision,
        #              eval_dest + '/recall': recall, eval_dest + '/F1': F1}
        # if return_logits:
        #     eval_dict[eval_dest + '/logits'] = y_logits
        eval_dict = {}
        return eval_dict
