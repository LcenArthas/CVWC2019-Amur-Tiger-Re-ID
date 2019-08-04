# encoding: utf-8
"""
@author:  loveletter
@contact: liucen0501@163.com
"""

import logging

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
from tensorboardX import  SummaryWriter

from utils.reid_metric import R1_mAP
import warnings

warnings.filterwarnings('ignore')

global ITER
ITER = 0

def create_supervised_trainer(model, optimizer, loss_fn,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        img, img_body, img_paw, target = batch

        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        img_body = img_body.to(device) if torch.cuda.device_count() >= 1 else img_body
        img_paw = img_paw.to(device) if torch.cuda.device_count() >= 1 else img_paw
        target = target.to(device) if torch.cuda.device_count() >= 1 else target

        score_g, score_gb, score_gp, global_feature, g_b_feature, g_p_feature = model(img, img_body, img_paw)
        id_g, id_gb, id_gp, triplet_gb, triplet_gp= loss_fn(score_g, score_gb, score_gp, g_b_feature, g_p_feature, target)
        loss_totle = id_g + 1.5 *id_gb + 1.5 * id_gp + 2 * triplet_gb + 2 * triplet_gp
        loss_totle.backward()
        optimizer.step()
        # compute acc
        acc = (score_g.max(1)[1] == target).float().mean()
        return loss_totle.item(), id_g.item(), id_gb.item(), id_gp.item(), triplet_gb.item(), triplet_gp.item(), acc.item()

    return Engine(_update)

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
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids, path = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, pids, camids, path

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

def create_summary_writer(model, data_loader):
    writer = SummaryWriter()
    data_loader_iter = iter(data_loader)
    img, img_body, img_paw, target = next(data_loader_iter)
    try:
        writer.add_graph(model, ((img, img_body, img_paw), ))
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer

def do_train(
        cfg,
        model,
        eval_model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    writer = create_summary_writer(model, train_loader)

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(eval_model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=1000, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model.state_dict(),
                                                                     'optimizer': optimizer.state_dict()})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss_totale')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_loss_id_g')
    RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'avg_loss_id_gb')
    RunningAverage(output_transform=lambda x: x[3]).attach(trainer, 'avg_loss_id_gp')
    RunningAverage(output_transform=lambda x: x[4]).attach(trainer, 'avg_loss_triple_gb')
    RunningAverage(output_transform=lambda x: x[5]).attach(trainer, 'avg_loss_triple_gp')
    RunningAverage(output_transform=lambda x: x[6]).attach(trainer, 'avg_acc')


    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1
        writer.add_scalar('training/loss', engine.state.metrics['avg_loss_totale'], engine.state.iteration)
        writer.add_scalar('training/loss_id_g', engine.state.metrics['avg_loss_id_g'], engine.state.iteration)
        writer.add_scalar('training/loss_id_gb', engine.state.metrics['avg_loss_id_gb'], engine.state.iteration)
        writer.add_scalar('training/loss_id_gp', engine.state.metrics['avg_loss_id_gp'], engine.state.iteration)
        writer.add_scalar('training/loss_triple_gb', engine.state.metrics['avg_loss_triple_gb'], engine.state.iteration)
        writer.add_scalar('training/loss_triple_gp', engine.state.metrics['avg_loss_triple_gp'], engine.state.iteration)
        writer.add_scalar('training/accuracy', engine.state.metrics['avg_acc'], engine.state.iteration)
        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss_totale'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP, PATH = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.3%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.3%}".format(r, cmc[r - 1]))

            writer.add_scalar("valdation MAP/avg_loss", mAP, engine.state.epoch)
            writer.add_scalar("valdation Rank-1/avg_accuracy", cmc[0], engine.state.epoch)
            writer.add_scalar("valdation Rank-5/avg_accuracy", cmc[4], engine.state.epoch)
            writer.add_scalar("valdation Rank-10/avg_accuracy", cmc[9], engine.state.epoch)

    trainer.run(train_loader, max_epochs=epochs)
    writer.close()