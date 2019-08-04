# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir

import torch
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model
from utils.logger import setup_logger
from data.datasets.eval_reid import eval_func
import shutil

import numpy as np
import json
from sklearn.preprocessing import normalize


def main(w):
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="configs/tiger.yml", help="path to config file", type=str
    )
    # parser.add_argument("opts", help="Modify config options using the command-line", default=None,
    #                     nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    cfg.MODEL.PRETRAIN_CHOICE = 'self'
    cfg.TEST.WEIGHT = w                                             #测试的模型
    cfg.MODEL.DEVICE = 'cuda'                                       #----------------->设定为cpu
    cfg.IS_DEMO = True                                              #设定为demo
    cfg.MODEL.DEVICE_ID='0'

    name1 = w.split('/')[-1].split('_')[0]
    name2 = w.split('/')[-1].split('_')[1]

    if name1 == 'se':
        cfg.MODEL.NAME = name1 + '_' + name2
        if '-' in name2:
            cfg.MODEL.BODYNAME = 'resnet34-bsize'
            cfg.INPUT.SIZE_TEST = [256, 512]
        else:
            cfg.MODEL.BODYNAME = 'resnet34'
            cfg.INPUT.SIZE_TEST = [128, 256]
    else:
        cfg.MODEL.NAME = name1
        if '-' in name1:
            cfg.MODEL.BODYNAME = 'resnet34-bsize'
            cfg.INPUT.SIZE_TEST = [256, 512]
        else:
            cfg.MODEL.BODYNAME = 'resnet34'
            cfg.INPUT.SIZE_TEST = [128, 256]

    print(cfg.MODEL.NAME)
    print(cfg.INPUT.SIZE_TEST)
    # cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    # logger = setup_logger("reid_baseline", output_dir, 0)
    # logger.info("Using {} GPUS".format(num_gpus))
    # logger.info(args)

    # if args.config_file != "":
    #     logger.info("Loaded configuration file {}".format(args.config_file))
    #     with open(args.config_file, 'r') as cf:
    #         config_str = "\n" + cf.read()
    #         logger.info(config_str)
    # logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)
    model, eval_model = build_model(cfg, num_classes)

    if cfg.MODEL.DEVICE == 'cuda':
        model.load_param(cfg.TEST.WEIGHT)
    else:
        model.load_param(cfg.TEST.WEIGHT, cpu=cfg.MODEL.DEVICE)

    return inference(cfg, eval_model, val_loader, num_query)                    #返回距离矩阵

if __name__ == '__main__':
    #多模型融合也K-FLOD
    root = './trained_weight/'
    #plain_re-id
    pic_num = 1764
    #wide_re-id

    model_pred = os.listdir(root)
    print(model_pred)

    num_model = len(model_pred)
    mat = np.zeros((pic_num, pic_num))
    q_path = np.empty((pic_num,))
    g_path = np.empty((pic_num,))

    for weight in model_pred:
        weight = root+ weight
        dismat, q_paths, g_paths = main(w=weight)

        #对得到的矩阵归一化处理
        dismat = normalize(dismat, axis=1, norm='l2')

        q_path = q_paths
        g_path = g_paths
        mat += dismat

    mat /= num_model
    PATHS = eval_func(distmat=mat, q_paths=q_path, g_paths=g_path, max_rank= pic_num, is_demo=True)

    print('#'*100)
    print('MAKE SUBMISSION.....')
    result = []
    for row in PATHS:
        r = {}
        r['query_id'] = int(row[-1].split('/')[-1].split('.')[0])
        ans_id = []
        for p in row[:-1]:
            ans_id.append(int(p.split('/')[-1].split('.')[0]))
        r['ans_ids'] = ans_id
        result.append(r)

    with open('submition_plain.json', 'w') as f:
        json.dump(result, f)

