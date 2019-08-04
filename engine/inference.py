# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import logging

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import Timer

from utils.reid_metric import R1_mAP, R1_mAP_reranking

import shutil
import os
import json

def create_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device == 'cuda':
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids, paths = batch
            # data = data.to(device) if torch.cuda.device_count() >= 1 else data
            data = data.to(device) if device=='cuda' else data
            feat = model(data)
            return feat, pids, camids, paths

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def inference(
        cfg,
        model,
        val_loader,
        num_query
):
    device = cfg.MODEL.DEVICE
    is_demo = cfg.IS_DEMO

    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Enter inferencing")
    if cfg.TEST.RE_RANKING == 'no':
        print("Create evaluator")
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, is_demo=cfg.IS_DEMO)},
                                                device=device)
    elif cfg.TEST.RE_RANKING == 'yes':
        print("Create evaluator for reranking")
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP_reranking(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, is_demo=cfg.IS_DEMO)},
                                                device=device)
    else:
        print("Unsupported re_ranking config. Only support for no or yes, but got {}.".format(cfg.TEST.RE_RANKING))

    timer = Timer(average=True)
    timer.attach(evaluator, start=Events.STARTED, pause=Events.COMPLETED)

    evaluator.run(val_loader)
    if is_demo:
        dismat, q_paths, g_paths = evaluator.state.metrics['r1_mAP']

        return dismat, q_paths, g_paths                                             #---------------
        #生成提交文件
        # result = []
        # for row in PATHS:
        #     r = {}
        #     r['query_id'] = row[-1].split('/')[-1]
        #     ans_id = []
        #     for p in row[:-1]:
        #         ans_id.append(p.split('/')[-1])
        #     r['ans_ids'] = ans_id
        #     result.append(r)
        #
        # with open('submition.json', 'w') as f:
        #     json.dump(result, f)

    else:
        cmc, mAP, PATHS = evaluator.state.metrics['r1_mAP']
        logger.info('Validation Results')
        logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        logger.info('use {:.3f}s'.format(timer.value()))

    #如果是demo，就把选出的图片放入新的文件夹
    # if is_demo:
    #     shutil.rmtree('./data/AmurTiger/demo_show/')
    #     os.mkdir('./data/AmurTiger/demo_show/')
    #     for i, file in enumerate(PATHS[0][:5]):
    #         new_name = file.split('/')[-1].split('_')[0] +'_'+ str(i) +'.jpg'
            # shutil.copyfile(file, './data/AmurTiger/demo_show/'+ str(1-score[i+1])[:5]+ '_' + new_name)
