import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from utils.box_utils import bbox_iou, xywh2xyxy, xyxy2xywh, generalized_box_iou
from utils.misc import get_world_size


def zmod_loss(outputs, bbox_mask, mass_mask):
    """Compute the losses related to the bounding boxes, 
       including the L1 regression loss and the GIoU loss
    """

    losses = {}
    box_score = outputs['box_score']
    seg_score = outputs['seg_score']
    img_mask = outputs['img_mask']

    loss_fuc = nn.BCEWithLogitsLoss()
    if box_score is not None:

        losses['box_loss'] = loss_fuc(box_score[-1], bbox_mask)

    if seg_score is not None:
        losses['seg_loss'] = loss_fuc(seg_score[-1], mass_mask)

    return losses


class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367."""

    def __init__(self, alpha=0.1, gamma=2.0):
        """Initialize the VarifocalLoss class."""
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, p, q):
        """Computes varfocal loss."""
        p_sigmoid = p.sigmoid()
        focal_weight = q * (q > 0.0).float() + \
                       self.alpha * (p_sigmoid - q).abs().pow(self.gamma) * \
                       (q <= 0.0).float()
        with torch.cuda.amp.autocast(enabled=True):
            loss = (F.binary_cross_entropy_with_logits(p.float(), q.float(), reduction='none') *
                    focal_weight).sum() / focal_weight.size(0)
        return loss
