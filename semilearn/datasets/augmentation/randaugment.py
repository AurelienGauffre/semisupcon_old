# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Code in this file is adapted from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
# This code is modified version of one of ildoonet, for randaugmentation of fixmatch.

import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# From lightly SimClr implem
from typing import Dict, List, Optional, Tuple, Union
import torchvision.transforms as T
from PIL.Image import Image
from torch import Tensor
from lightly.transforms.gaussian_blur import GaussianBlur
from lightly.transforms.multi_view_transform import MultiViewTransform
from lightly.transforms.rotation import random_rotation_transform
from lightly.transforms.utils import IMAGENET_NORMALIZE


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Identity(img, v):
    return img


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v):  # [-30, 30]
    #assert -30 <= v <= 30
    #if random.random() > 0.5:
    #    v = -v
    return img.rotate(v)



def Sharpness(img, v):  # [0.1,1.9]
    assert v >= 0.0
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v):  # [-0.3, 0.3]
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    #assert v >= 0.0
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    #assert -0.3 <= v <= 0.3
    #if random.random() > 0.5:
    #    v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    #assert 0 <= v
    #if random.random() > 0.5:
    #    v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def Cutout(img, v):  #[0, 60] => percentage: [0, 0.2] => change to [0, 0.5]
    assert 0.0 <= v <= 0.5
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img

    
def augment_list():  
    l = [
        (AutoContrast, 0, 1),
        (Brightness, 0.05, 0.95),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Equalize, 0, 1),
        (Identity, 0, 1),
        (Posterize, 4, 8),
        (Rotate, -30, 30),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.3, 0.3),
        (ShearY, -0.3, 0.3),
        (Solarize, 0, 256),
        (TranslateX, -0.3, 0.3),
        (TranslateY, -0.3, 0.3)
    ]
    return l

def augment_list_no_color():  
    l = [
        # (AutoContrast, 0, 1),
        (Brightness, 0.05, 0.95),
        # (Color, 0.05, 0.95),
        # (Contrast, 0.05, 0.95),
        (Equalize, 0, 1),
        (Identity, 0, 1),
        # (Posterize, 4, 8),
        (Rotate, -30, 30),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.3, 0.3),
        (ShearY, -0.3, 0.3),
        # (Solarize, 0, 256),
        (TranslateX, -0.3, 0.3),
        (TranslateY, -0.3, 0.3)
    ]
    return l

    
class RandAugment:
    def __init__(self, n, m, exclude_color_aug=False):
        self.n = n
        self.m = m      # [0, 30] in fixmatch, deprecated.
        if not exclude_color_aug:
            self.augment_list = augment_list()
        else:
            self.augment_list = augment_list_no_color()


        
    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val)*random.random()
            img = op(img, val) 
        cutout_val = random.random() * 0.5 
        img = Cutout(img, cutout_val) #for fixmatch
        return img



############################################################################## SimCLR from Lightly
### We are just adding cutout to the SimCLR augmentations to make it fair

class CutOutTransform:
    def __init__(self,):
        v = random.random() * 0.5
        self.v = v

    def __call__(self, img):
        return Cutout(img, self.v)


class SimCLRTransformAndCutout:
    def __init__(
        self,
        input_size: int = 224,
        cj_prob: float = 0.8,
        cj_strength: float = 1.0,
        cj_bright: float = 0.8,
        cj_contrast: float = 0.8,
        cj_sat: float = 0.8,
        cj_hue: float = 0.2,
        min_scale: float = 0.5,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.5,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Union[None, float, Tuple[float, float]] = None,
        normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
    ):
        color_jitter = T.ColorJitter(
            brightness=cj_strength * cj_bright,
            contrast=cj_strength * cj_contrast,
            saturation=cj_strength * cj_sat,
            hue=cj_strength * cj_hue,
        )

        transform = [
            T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
            random_rotation_transform(rr_prob=rr_prob, rr_degrees=rr_degrees),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),
            CutOutTransform(),
            GaussianBlur(kernel_size=kernel_size, sigmas=sigmas, prob=gaussian_blur),
            # T.ToTensor(),# Pas de to tensor pour cette librairie
        ]
        # if normalize:
        #     transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])] #normalization is done in dataset
        self.transform = T.Compose(transform)

    def __call__(self, image: Union[Tensor, Image]) -> Tensor:
        """
        Applies the transforms to the input image.

        Args:
            image:
                The input image to apply the transforms to.

        Returns:
            The transformed image.

        """
        transformed: Tensor = self.transform(image)
        return transformed
    
if __name__ == '__main__':
    # randaug = RandAugment(3,5)
    # print(randaug)
    # for item in randaug.augment_list:
    #     print(item)
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    img = PIL.Image.open('./u.jpg')
    randaug = RandAugment(3,6)
    simclr = SimCLRTransformAndCutout()
    img = randaug(img)
    #img = simclr(img)
    import matplotlib
    from matplotlib import pyplot as plt 
    plt.imshow(img)
    plt.show()

