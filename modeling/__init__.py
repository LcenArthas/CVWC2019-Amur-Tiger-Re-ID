# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .glabole_stream import Glabole_stream
from .part_stream import Part_body_stream, Part_paw_stream
from .pipline import Pipline

def build_model(cfg, num_classes):
    # if cfg.MODEL.NAME == 'resnet50':
    #     model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT)
    glabole_model = Glabole_stream(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)
    part_body_model = Part_body_stream(cfg.MODEL.PRETRAIN_BODYPATH, cfg.MODEL.BODYNAME, cfg.MODEL.PRETRAIN_CHOICE)
    part_paw_model = Part_paw_stream(cfg.MODEL.PRETRAIN_BODYPATH, cfg.MODEL.BODYNAME, cfg.MODEL.PRETRAIN_CHOICE)
    model = Pipline(glabole_model, part_body_model, part_paw_model, num_classes)
    return model, glabole_model
