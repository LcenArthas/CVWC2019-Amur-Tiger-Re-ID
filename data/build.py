# encoding: utf-8
"""
@author:  letter
@contact: liucen05@163.com
"""

from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, val_collate_fn
from .datasets import init_dataset, ImageDataset
from .samplers import RandomIdentitySampler
from .transforms import build_transforms


def make_data_loader(cfg):
    train_transforms_globale, train_transforms_body, train_transforms_paw = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    is_demo = cfg.IS_DEMO

    if is_demo:
        dataset = init_dataset(cfg.DATASETS.NAMES, index_flod=cfg.DATASETS.INDEX_FLOD,
                               root=cfg.DATASETS.ROOT_DIR, is_demo=True)  # Return a class：Market1501
    else:
        dataset = init_dataset(cfg.DATASETS.NAMES, index_flod=cfg.DATASETS.INDEX_FLOD,
                               root=cfg.DATASETS.ROOT_DIR)                # Return a class：Market1501

    num_classes = dataset.num_train_pids
    train_set = ImageDataset(cfg, dataset.train, transform_g=train_transforms_globale,
                             transform_body=train_transforms_body,transform_paw=train_transforms_paw, is_train=True)
    if cfg.DATALOADER.SAMPLER == 'softmax':
        #这是没有三元组的时候
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        #这是有三元组的时候
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn
        )

    #注意这里查询集和被查询集链接
    val_set = ImageDataset(cfg, dataset.query + dataset.gallery, transform_g=val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, val_loader, len(dataset.query), num_classes
