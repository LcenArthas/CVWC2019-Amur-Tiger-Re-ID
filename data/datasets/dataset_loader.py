# encoding: utf-8
"""
@author:  loveletter
@contact: liucen05@163.com
"""

import os.path as osp
from PIL import Image, ImageFilter
import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as T
from imgaug import augmenters as iaa

# from data.datasets.cut_mask import body_mask, paw_mask


def read_image(img_path, partsize=0):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    if img_path == 'no':
        h = partsize[1]
        w = partsize[0]
        img = np.zeros((h, w, 3), np.uint8)
        return Image.fromarray(img, mode='RGB')

    else:
        got_img = False
        if not osp.exists(img_path):
            raise IOError("{} does not exist".format(img_path))
        while not got_img:
            try:
                img = Image.open(img_path).convert('RGB')
                got_img = True
            except IOError:
                print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
                pass
        return img

def amaugimg(image):
    #数据增强
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    seq = iaa.Sequential([
        # iaa.Affine(rotate=(-5, 5),
        #            shear=(-5, 5),
        #            mode='edge'),

        iaa.SomeOf((0, 2),                        #选择数据增强
                   [
                       iaa.GaussianBlur((0, 1.5)),
                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01 * 255), per_channel=0.5),
                       # iaa.AddToHueAndSaturation((-5, 5)),  # change hue and saturation
                       iaa.PiecewiseAffine(scale=(0.01, 0.03)),
                       iaa.PerspectiveTransform(scale=(0.01, 0.1))
                   ],
                   random_order=True
                   )
    ])
    image = seq.augment_image(image)

    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return image

class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, cfg, dataset, transform_g=None, transform_body=None, transform_paw=None,is_train=False):
        self.dataset = dataset
        self.transform = transform_g
        self.transform_body = transform_body
        self.transform_paw = transform_paw
        self.is_train = is_train
        self.input_size = cfg.INPUT.SIZE_TRAIN
        self.part_bodysize = cfg.PART.SIZE_BODY
        self.part_pawsize = cfg.PART.SIZE_PAW

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.is_train:
            img_path, pid, camid, body, p1, p2, p3, p4, p5, p6 = self.dataset[index]
            img = read_image(img_path)  # 读图

            img_body = read_image(body, self.part_bodysize)
            img_p1 = read_image(p1, self.part_pawsize)
            img_p2 = read_image(p2, self.part_pawsize)
            img_p3 = read_image(p3, self.part_pawsize)
            img_p4 = read_image(p4, self.part_pawsize)
            img_p5 = read_image(p5, self.part_pawsize)
            img_p6 = read_image(p6, self.part_pawsize)

            img = amaugimg(img)
            img_body = amaugimg(img_body)
            img_p1 = amaugimg(img_p1)
            img_p2 = amaugimg(img_p2)
            img_p3 = amaugimg(img_p3)
            img_p4 = amaugimg(img_p4)
            img_p5 = amaugimg(img_p5)
            img_p6 = amaugimg(img_p6)

            if self.transform is not None:
                img = self.transform(img)
                img_body = self.transform_body(img_body)
                img_p1 = self.transform_paw(img_p1)
                img_p2 = self.transform_paw(img_p2)
                img_p3 = self.transform_paw(img_p3)
                img_p4 = self.transform_paw(img_p4)
                img_p5 = self.transform_paw(img_p5)
                img_p6 = self.transform_paw(img_p6)
            img_parts = torch.cat((img_p1, img_p2, img_p3, img_p4, img_p5, img_p6), dim=0)           #在通道方向拼接parts
            return img, img_body, img_parts ,pid, camid

        else:
            img_path, pid, camid,  = self.dataset[index]
            img = read_image(img_path)                    #读图

            if self.transform is not None:
                img = self.transform(img)
            return img, pid, camid, img_path
