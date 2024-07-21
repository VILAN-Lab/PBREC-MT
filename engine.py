# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import torch
import torch.distributed as dist

from tqdm import tqdm
from typing import Iterable

import utils.misc as utils
import utils.loss_utils as loss_utils
import utils.eval_utils as eval_utils

import numpy as np

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    scaler=None, scheduler=None):

    # model = torch.compile(model)

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for batch, it in metric_logger.log_every(data_loader, print_freq, header):

        img_data, text_data, _, _, mass_mask, bbox_mask = batch
        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        # target = target.to(device)
        mass_mask = mass_mask.to(device)
        bbox_mask = bbox_mask.to(device)

        # model forward
        output = model(img_data, text_data)

        loss_dict = loss_utils.zmod_loss(output, bbox_mask, mass_mask)
        losses = sum(loss_dict[k] for k in loss_dict.keys())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {k: v for k, v in loss_dict_reduced.items()}
        losses_reduced_unscaled = sum(loss_dict_reduced_unscaled.values())
        loss_value = losses_reduced_unscaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is None:
            losses.backward()
            optimizer.step()
        else:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        metric_logger.update(loss=loss_value, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(model: torch.nn.Module, data_loader: Iterable, device: torch.device, amp: bool = False):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Eval:'

    for batch, i in metric_logger.log_every(data_loader, 10, header):
        img_data, text_data, target, gt_mask, mass_mask, bbox_mask = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)
        mass_mask = mass_mask.to(device)
        bbox_mask = bbox_mask.to(device)
        
        outputs = model(img_data, text_data)
        if outputs['box_score'] is not None:
            accu = eval_utils.zmod_eval_val(outputs['box_score'][-1], bbox_mask > 0)
        else:
            accu = eval_utils.zmod_eval_val(outputs['seg_score'][-1], mass_mask > 0)
        # iou_value = torch.mean(iou).item()
        accu_value = accu.item()
        # metric_logger.update_v2('iou', iou_value, batch_size)
        metric_logger.update_v2('accu', accu_value, batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    box_score_list = []
    seg_score_list = []
    img_mask_list = []
    bbox_mask_list = []
    gt_box_list = []
    gt_seg_list = []
    for _, batch in enumerate(tqdm(data_loader)):
        img_data, text_data, target, gt_mask, mass_mask, bbox_mask = batch
        batch_size = img_data.tensors.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)

        output = model(img_data, text_data)
        if 'box' in model.task_type:
            box_score_list.append(output['box_score'][-1].detach().cpu())
        if 'seg' in model.task_type:
            seg_score_list.append(output['seg_score'][-1].detach().cpu())
        img_mask_list.append(img_data.mask.flatten(1).detach().cpu())
        bbox_mask_list.append(bbox_mask.cpu())
        gt_box_list.append(target.cpu())
        gt_seg_list.append(gt_mask.cpu())

    box_score = None
    if 'box' in model.task_type:
        box_score = torch.cat(box_score_list, dim=0).data.numpy()
    seg_score = None
    if 'seg' in model.task_type:
        seg_score = torch.cat(seg_score_list, dim=0).data.numpy()
    img_mask = torch.cat(img_mask_list, dim=0).data.to(torch.bool)
    gt_box = torch.cat(gt_box_list, dim=0).data
    gt_seg = torch.cat(gt_seg_list, dim=0).data.to(torch.bool)
    bbox_mask = torch.cat(bbox_mask_list, dim=0)
    total_num = bbox_mask.shape[0]
    # accu_num = eval_utils.zmod_eval_test(box_score, bbox_mask > 0)
    # result_tensor = torch.tensor([accu_num, total_num]).to(device)
    # accuracy = float(result_tensor[0]) / float(result_tensor[1])
    accuracy = 0
    return gt_box, gt_seg, box_score, seg_score, img_mask, accuracy
        