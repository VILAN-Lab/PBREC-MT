import torch
from tqdm import tqdm
import os
import numpy as np
from utils.box_utils import xywh2xyxy, xyxy2xywh
import torch.nn.functional as F
import datasets.transforms as T
from PIL import Image


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    # print(box1, box1.shape)
    # print(box2, box2.shape)
    return inter_area / (b1_area + b2_area - inter_area + 1e-16)


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

    # print(box1, box1.shape)
    # print(box2, box2.shape)
    return inter_area / (b_area.unsqueeze(0) + b_area.unsqueeze(-1) - inter_area + 1e-16)


def get_mask(gt):
    gt = xyxy2xywh(gt)
    gt_fuse = gt.squeeze(0) / 32
#     gt_fuse = xyxy2xywh(gt_fuse)
    anchor = torch.cat([torch.arange(20).unsqueeze(-1).repeat(1, 20).view(20, 20, 1) + 0.5,
                        torch.arange(20).unsqueeze(0).repeat(20, 1).view(20, 20, 1) + 0.5], dim=-1)
    bbox_mask = torch.zeros([20, 20])
    bbox_mask[int(gt_fuse[1]) - 1: int(gt_fuse[1]) + 2, int(gt_fuse[0]) - 1: int(gt_fuse[0]) + 2] = 1
    lt, rb = torch.clip(anchor - torch.floor(gt_fuse[:2] - gt_fuse[2:] / 2), min=0), torch.clip(torch.ceil(gt_fuse[:2] + gt_fuse[2:] / 2) - anchor, min=0)
    center_mask = torch.sqrt((torch.min(lt[:, :, 0], rb[:, :, 0]) / torch.max(lt[:, :, 0], rb[:, :, 0])) *
                             (torch.min(lt[:, :, 1], rb[:, :, 1]) / torch.max(lt[:, :, 1], rb[:, :, 1]))).T
    center_mask = (center_mask * bbox_mask / center_mask.max() > 0).to(torch.int)
    center_mask = center_mask.flatten(0)
    indices = torch.nonzero(center_mask == 1)
#     print(bbox[indices.squeeze(-1),:])
    return indices.squeeze(-1)


if __name__ == '__main__':
    center = torch.cat([torch.arange(20).unsqueeze(-1).repeat(1, 20).view(20, 20, 1) + 0.5,
                        torch.arange(20).unsqueeze(0).repeat(20, 1).view(20, 20, 1) + 0.5], dim=-1).transpose(0, 1) * 32
    center = center.permute(2, 0, 1).flatten(1).transpose(0, 1)

    bbox_dict = torch.load(r'./fcos_r101.pth')

    path_list = [r'gref_umd/gref_umd_val',
                 r'gref_umd/gref_umd_test',
                 r'gref/gref_val',
                 r'unc/unc_val',
                 r'unc/unc_testA',
                 r'unc/unc_testB',
                 r'unc+/unc+_val',
                 r'unc+/unc+_testA',
                 r'unc+/unc+_testB']
    n = 0
    root = '.'
    dataset_path = 'data'
    out_path = 'out'
    data = torch.load(f'{os.path.join(dataset_path, path_list[n])}.pth')
    gt_dict = torch.load('./multitask/gref_umd_val.pth')
    # grad_cam = torch.load('grad_cam.pth')
    gt_list = xywh2xyxy(torch.Tensor(gt_dict['gt_box'])) * 640
    box_score_list = gt_dict['box_score']
    mask_list = gt_dict['img_mask']
    # box_score_list = grad_cam
    # seg_score_list = gt_dict['seg_score']

    # transform = T.Compose([T.RandomResize([640]), T.ToTensor(), T.NormalizeAndPad(size=640)])

    acc_num = 0
    for i, d in tqdm(enumerate(data)):
        #     out = np.load(f'{os.path.join(out_path, d[0][: -4])}.npz')
        #     bbox = out['bbox']
        bbox = torch.tensor(bbox_dict[d[0]]['box'])
        # img = Image.open(f'./ln_data/other/images/mscoco/images/train2014/{d[0]}').convert("RGB")
        # img_mask = transform({'img': img, 'box':  torch.tensor([1, 2, 3, 4], dtype=int)})['mask']
        # img_mask = (F.interpolate(img_mask[None].unsqueeze(0).float(), size=bbox.shape[-2:]).view(400) == 0)
        conf = F.sigmoid(torch.tensor(bbox_dict[d[0]]['center'])).view(400)
        cls_score = F.softmax(torch.tensor(bbox_dict[d[0]]['cls']).view(80, 400), dim=0)
        cls_score, cls_index = torch.max(cls_score, dim=0)
        # cls = torch.argmax(torch.tensor(bbox_dict[d[0]]['cls']), dim=1).view(400)
        bbox = bbox.squeeze(0).flatten(1).transpose(0, 1)
        bbox[:, 0] = center[:, 0] - bbox[:, 0]
        bbox[:, 1] = center[:, 1] - bbox[:, 1]
        bbox[:, 2] = center[:, 0] + bbox[:, 2]
        bbox[:, 3] = center[:, 1] + bbox[:, 3]
        bbox = bbox.clip(0, 640)
        gt = gt_list[i]
        # center_score = F.sigmoid(torch.tensor(box_score_list[i]))
        center_score = F.sigmoid(torch.tensor(box_score_list[i]))
        mask = torch.tensor(mask_list[i])
        # mask = get_mask(gt)
        #     bbox = xywh2xyxy(bbox)
        #     gt = xywh2xyxy(gt * 640)
        #     image_path = os.path.join('..','refer/data/images/mscoco/images/train2014', d[0])
        #     img = cv2.imread(image_path)
        #     l, t, r, b = gt[0].int().numpy()
        #     cv2.rectangle(img,(l, t),(r, b),(0,255,0),3)
        #     cv2.imshow('img',img)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        # _, indices = torch.topk(center_score, k=16)
        # indices = torch.nonzero(torch.sqrt(center_score * conf) >= 0.3).squeeze()
        _, indices = torch.topk(center_score * conf, k=12)
        # _, indices = torch.topk(center_score, k=12)
        bbox_k, conf_k, cls_k = bbox[indices].view(-1, 4), conf[indices], cls_index[indices]
        # bbox_rank1 = bbox[indices[0]]
        # pre_iou = bbox_iou(bbox_rank1.unsqueeze(0), bbox_k)
        # indices = torch.nonzero(cls_k == cls[indices[0]]).squeeze()
        # torch.sum(iou_matrix(bbox_k).fill_diagonal_(0) * conf_k, dim=0) * conf_k
        rank = torch.sum(iou_matrix(bbox_k).fill_diagonal_(0), dim=0)
               # (torch.sum((cls_k.unsqueeze(0) == cls_k.unsqueeze(1)), dim=0) - 1)
        pre = bbox_k[torch.argmax(rank)].view(-1, 4)
        # pre = bbox_k[0].view(-1, 4)
        # indices = indices[torch.argmax(conf[indices])]  # (rank+center-ness)top-1
        iou = bbox_iou(gt.unsqueeze(0), pre)
        acc_num += (torch.sum(iou >= 0.3)) > 0
    print(acc_num / len(data))
    # for path in path_list:
    #     data = torch.load(f'{os.path.join(dataset_path, path)}.pth')
    #     for d in data:
    #         image_path = os.path.join('..','refer/data/images/mscoco/images/train2014', d[0])
    #         img = cv2.imread(image_path)
    #         cv2.imwrite(r'./image/'+path + '/' + d[0], img)