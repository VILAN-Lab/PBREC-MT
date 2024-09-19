import torch
from tqdm import tqdm
import os
import numpy as np
from utils.box_utils import xywh2xyxy, xyxy2xywh
import torch.nn.functional as F
import datasets.transforms as T


center = torch.cat([torch.arange(20).unsqueeze(-1).repeat(1, 20).view(20, 20, 1) + 0.5,
                        torch.arange(20).unsqueeze(0).repeat(20, 1).view(20, 20, 1) + 0.5], dim=-1).transpose(0, 1) * 32
center = center.permute(2, 0, 1).flatten(1).transpose(0, 1)


def center2xyxy(x):
    x = x.squeeze(0).flatten(1).transpose(0, 1)
    x[:, 0] = center[:, 0] - x[:, 0]
    x[:, 1] = center[:, 1] - x[:, 1]
    x[:, 2] = center[:, 0] + x[:, 2]
    x[:, 3] = center[:, 1] + x[:, 3]
    x = x.clip(0, 640)
    return x


def iou_matrix(box, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b_x1, b_y1, b_x2, b_y2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b_x1, b_x2 = box[:, 0] - box[:, 2] / 2, box[:, 0] + box[:, 2] / 2
        b_y1, b_y2 = box[:, 1] - box[:, 3] / 2, box[:, 1] + box[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b_x1.unsqueeze(0), b_x1.unsqueeze(-1))
    inter_rect_y1 = torch.max(b_y1.unsqueeze(0), b_y1.unsqueeze(-1))
    inter_rect_x2 = torch.min(b_x2.unsqueeze(0), b_x2.unsqueeze(-1))
    inter_rect_y2 = torch.min(b_y2.unsqueeze(0), b_y2.unsqueeze(-1))
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b_area = (b_x2 - b_x1) * (b_y2 - b_y1)

    return inter_area / (b_area.unsqueeze(0) + b_area.unsqueeze(-1) - inter_area + 1e-16)


def post_process_bbox(gt_data, od_data, pred_score, gt_box, k=12):

    acc_num = 0
    for i, d in tqdm(enumerate(gt_data)):
        bbox = torch.tensor(od_data[d[0]]['box']).view(4, 400)
        bbox = center2xyxy(bbox)
        conf = F.sigmoid(torch.tensor(od_data[d[0]]['center'])).view(400)
        cls_score = F.softmax(torch.tensor(od_data[d[0]]['cls']).view(80, 400), dim=0)
        cls_score, cls_index = torch.max(cls_score, dim=0)

        gt = gt_box[i]

        center_score = F.sigmoid(torch.tensor(pred_score[i]))

        _, indices = torch.topk(center_score * conf, k=k)
        bbox_k, conf_k, cls_k = bbox[indices].view(-1, 4), conf[indices], cls_index[indices]
        rank = torch.sum(iou_matrix(bbox_k).fill_diagonal_(0), dim=0)
        pre = bbox_k[torch.argmax(rank)].view(-1, 4)
        iou = bbox_iou(gt.unsqueeze(0), pre)
        acc_num += (torch.sum(iou >= 0.5)) > 0
        
    return acc_num / len(gt_data)
