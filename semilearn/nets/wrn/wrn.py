# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from semilearn.nets.utils import load_checkpoint

momentum = 0.001


def mish(x):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function (https://arxiv.org/abs/1908.08681)"""
    return x * torch.tanh(F.softplus(x))


class PSBatchNorm2d(nn.BatchNorm2d):
    """How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/2001.11216)"""

    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x):
        return super().forward(x) + self.alpha


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, first_stride, num_classes, depth=28, widen_factor=2, drop_rate=0.0, **kwargs):
        super(WideResNet, self).__init__()
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=True)
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, first_stride, drop_rate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate)

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001, eps=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.channels = channels[3]
        self.num_features = channels[3]

        self.classifier = nn.Linear(channels[3], num_classes)  # proba for BCE

        self.contrastive_head = nn.Sequential(  # projection for contrastive loss
            nn.Linear(channels[3], channels[3]),
            nn.ReLU(inplace=True),
            nn.Linear(channels[3], channels[3])
        )

        # rot_classifier for Remix Match
        # self.is_remix = is_remix
        # if is_remix:
        #     self.rot_classifier = nn.Linear(self.channels, 4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x, only_fc=False, only_feat=False, contrastive=False, detach_ce=False, **kwargs):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
            contrastive: bool parameter, if True, return both logits and projected features
                (instead of features)
        """

        if only_fc:
            return self.classifier(x)

        out = self.extract(x)
        out = F.adaptive_avg_pool2d(out, 1)
        feat = out.view(-1, self.channels)

        if only_feat:
            return feat

        if not contrastive:  # comme fait originalement
            logits = self.classifier(feat)
            result_dict = {'logits': logits, 'feat': feat}
            return result_dict
        else:  # puisque qu'on a besoin de .detach dans le training, on retourne que les features et les logits/contrastive proj sont obtenus dans la training step

            contrastive = self.contrastive_head(feat)

            logits = self.classifier(feat) if not detach_ce else self.classifier(feat.detach())

            result_dict = {'logits': logits, 'contrastive_feats': contrastive}
            return result_dict

    def extract(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        return out

    def group_matcher(self, coarse=False, prefix=''):
        matcher = dict(stem=r'^{}conv1'.format(prefix),
                       blocks=r'^{}block(\d+)'.format(prefix) if coarse else r'^{}block(\d+)\.layer.(\d+)'.format(
                           prefix))
        return matcher

    def no_weight_decay(self):
        nwd = []
        for n, _ in self.named_parameters():
            if 'bn' in n or 'bias' in n:
                nwd.append(n)
        return nwd


def wrn_28_2(pretrained=False, pretrained_path=None, droprate=0.0, **kwargs):
    model = WideResNet(first_stride=1, depth=28, widen_factor=2, droprate=droprate, **kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model


def wrn_28_8(pretrained=False, pretrained_path=None, droprate=0.0, **kwargs):
    model = WideResNet(first_stride=1, depth=28, widen_factor=8, droprate=droprate, **kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained_path)
    return model
