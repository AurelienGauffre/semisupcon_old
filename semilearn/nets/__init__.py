# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .resnet import resnet50, resnet18, resnet50_proto, resnet18_proto
from .wrn import wrn_28_2, wrn_28_8, wrn_var_37_2, wrn_28_2_proto, wrn_28_8_proto, wrn_var_37_2_proto
from .vit import vit_base_patch16_224, vit_small_patch16_224, vit_small_patch2_32, vit_tiny_patch2_32, vit_base_patch16_96
from .vit import vit_base_patch16_224_proto,    vit_small_patch16_224_proto,    vit_small_patch2_32_proto,    vit_tiny_patch2_32_proto,    vit_base_patch16_96_proto
from .bert import bert_base_cased, bert_base_uncased
from .wave2vecv2 import wave2vecv2_base
from .hubert import hubert_base
