import torch
import numpy as np

from utils.box_utils import bbox_iou, xywh2xyxy


def zmod_eval_val(pred_boxes, gt_masks):
    batch_size = pred_boxes.shape[0]
    # pred_indexes = torch.argmax(pred_boxes, dim=-1)
    # gt_indexes = torch.argmax(gt_masks, dim=-1)
    accu = torch.sum((pred_boxes == pred_boxes.max(dim=-1).values.unsqueeze(-1)) * gt_masks) / float(batch_size)

    # pred_indexes = torch.argmax(pred_boxes, dim=-1)
    # gt_indexes = torch.argmax(gt_masks, dim=-1)
    # accu = torch.sum(pred_indexes == gt_indexes) / float(batch_size)

    return accu

def zmod_eval_test(pred_boxes, gt_masks):
    accu_num = accu = torch.sum((pred_boxes == pred_boxes.max(dim=-1).values.unsqueeze(-1)) * gt_masks)
    # accu_num = iou >= 0.5

    return accu_num
