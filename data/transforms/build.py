# encoding: utf-8
"""
@author:  loveletter
@contact: liucen05@163.com
"""

import torchvision.transforms as T

from .transforms import RandomErasing


def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        transform_ = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomRotation(cfg.INPUT.RO_DEGREE),
            T.ColorJitter(brightness=cfg.INPUT.BRIGHT_PROB, saturation=cfg.INPUT.SATURA_PROB, contrast=cfg.INPUT.CONTRAST_PROB, hue=cfg.INPUT.HUE_PROB),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            normalize_transform
        ])
        transform_body = T.Compose([
            T.Resize(cfg.PART.SIZE_BODY),
            T.RandomRotation(cfg.INPUT.RO_DEGREE),
            T.ColorJitter(brightness=cfg.INPUT.BRIGHT_PROB, saturation=cfg.INPUT.SATURA_PROB,
                          contrast=cfg.INPUT.CONTRAST_PROB, hue=cfg.INPUT.HUE_PROB),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.PART.SIZE_BODY),
            T.ToTensor(),
            normalize_transform
        ])
        transform_paw = T.Compose([
            T.Resize(cfg.PART.SIZE_PAW),
            T.RandomRotation(cfg.INPUT.RO_DEGREE),
            T.ColorJitter(brightness=cfg.INPUT.BRIGHT_PROB, saturation=cfg.INPUT.SATURA_PROB,
                          contrast=cfg.INPUT.CONTRAST_PROB, hue=cfg.INPUT.HUE_PROB),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.PART.SIZE_PAW),
            T.ToTensor(),
            normalize_transform
        ])
        return transform_, transform_body, transform_paw
    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])
        return transform
