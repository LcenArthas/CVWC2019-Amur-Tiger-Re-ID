# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch


def train_collate_fn(batch):
    imgs, img_body, img_parts, pids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)

    return torch.stack(imgs, dim=0), torch.stack(img_body, dim=0), torch.stack(img_parts, dim=0),pids


def val_collate_fn(batch):
    imgs, pids, camids, paths = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, paths
