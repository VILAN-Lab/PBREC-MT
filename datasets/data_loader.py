# -*- coding: utf-8 -*-

"""
ReferIt, UNC, UNC+ and GRef referring image segmentation PyTorch dataset.

Define and group batches of images, segmentations and queries.
Based on:
https://github.com/chenxi116/TF-phrasecut-public/blob/master/build_batches.py
"""

import os
import re
import cv2
import sys
import json
import torch
import numpy as np
import os.path as osp
import scipy.io as sio
import torch.utils.data as data
from pycocotools import mask as cocomask

sys.path.append('.')

from PIL import Image
from pytorch_pretrained_bert.tokenization import BertTokenizer
from utils.box_utils import xywh2xyxy
import torch.nn.functional as F


def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line  # reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples


## Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


class DatasetNotFoundError(Exception):
    pass


class PBRECDataset(data.Dataset):
    SUPPORTED_DATASETS = {
        'referit': {'splits': ('train', 'val', 'trainval', 'test')},
        'unc': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco', 'split_by': 'unc'}
        },
        'unc+': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
        },
        'gref': {
            'splits': ('train', 'val'),
            'params': {'dataset': 'refcocog', 'split_by': 'google'}
        },
        'gref_umd': {
            'splits': ('train', 'val', 'test', 'adv', 'easy', 'hard'),
            'params': {'dataset': 'refcocog', 'split_by': 'umd'}
        },
        'flickr': {
            'splits': ('train', 'val', 'test')}
    }

    def __init__(self, data_root, split_root='data', dataset='referit',
                 transform=None, return_idx=False, testmode=False,
                 split='train', max_query_len=128,
                 bert_model='bert-base-uncased'):
        self.images = []
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.query_len = max_query_len
        self.transform = transform
        self.testmode = testmode
        self.split = split
        # self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.tokenizer = BertTokenizer.from_pretrained('./bert/bert-base-uncased-vocab.txt', do_lower_case=True)
        self.return_idx = return_idx

        assert self.transform is not None

        if split == 'train':
            self.augment = True
        else:
            self.augment = False

        if self.dataset == 'referit':
            self.dataset_root = osp.join(self.data_root, 'referit')
            self.im_dir = osp.join(self.dataset_root, 'images')
            self.split_dir = osp.join(self.dataset_root, 'splits')
        elif self.dataset == 'flickr':
            self.dataset_root = osp.join(self.data_root, 'Flickr30k')
            self.im_dir = osp.join(self.dataset_root, 'flickr30k_images')
        else:  ## refcoco, etc.
            self.dataset_root = osp.join(self.data_root, 'other')
            self.im_dir = osp.join(
                self.dataset_root, 'images', 'mscoco', 'images', 'train2014')
            self.split_dir = osp.join(self.dataset_root, 'splits')

        if not self.exists_dataset():
            # self.process_dataset()
            print('Please download index cache to data folder: \n \
                https://drive.google.com/open?id=1cZI562MABLtAzM6YU4WmKPFFguuVr0lZ')
            exit(0)

        dataset_path = osp.join(self.split_root, self.dataset)
        valid_splits = self.SUPPORTED_DATASETS[self.dataset]['splits']

        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self.dataset, split))

        splits = [split]
        if self.dataset != 'referit':
            splits = ['train', 'val'] if split == 'trainval' else [split]
        for split in splits:
            imgset_file = '{0}_{1}.pth'.format(self.dataset, split)
            imgset_path = osp.join(dataset_path, imgset_file)
            self.images += torch.load(imgset_path)

    def exists_dataset(self):
        return osp.exists(osp.join(self.split_root, self.dataset))

    def pull_item(self, idx):
        if self.dataset == 'flickr':
            img_file, bbox, phrase, r1_word = self.images[idx]
        else:
            img_file, _, bbox, phrase, attri, seg = self.images[idx]
            r1_word = attri[0][1][0]
        ## box format: to x1y1x2y2
        if not (self.dataset == 'referit' or self.dataset == 'flickr'):
            bbox = np.array(bbox, dtype=int)
            bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]
        else:
            bbox = np.array(bbox, dtype=int)

        img_path = osp.join(self.im_dir, img_file)
        img = Image.open(img_path).convert("RGB")
        # img = cv2.imread(img_path)
        # ## duplicate channel if gray image
        # if img.shape[-1] > 1:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # else:
        #     img = np.stack([img] * 3)

        bbox = torch.tensor(bbox, dtype=torch.float)
        seg = torch.tensor(seg[0], dtype=torch.float)
        return img, phrase, bbox, r1_word, seg

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, phrase, bbox, r1_word, seg = self.pull_item(idx)
        # phrase = phrase.decode("utf-8").encode().lower()
        phrase = phrase.lower()
        r1_word = r1_word.lower()
        input_dict = {'img': img, 'box': bbox, 'text': phrase, 'subj': r1_word, 'seg': seg}
        input_dict = self.transform(input_dict)
        img = input_dict['img']
        bbox = input_dict['box']
        seg = input_dict['seg'].numpy()
        phrase = input_dict['text']
        img_mask = input_dict['mask']
        ## encode phrase to bert input
        examples = read_examples(phrase, idx)
        features = convert_examples_to_features(
            examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
        word_id = features[0].input_ids
        word_mask = features[0].input_mask

        # bbox_xyxy = (xywh2xyxy(bbox) * 20).to(torch.int)
        bbox_patch = bbox * 20
        bbox_mask = torch.zeros([20, 20])
        bbox_mask[int(bbox_patch[1]) - 1: int(bbox_patch[1]) + 2, int(bbox_patch[0]) - 1: int(bbox_patch[0]) + 2] = 1
        # bbox_wh = [bbox_xyxy[2] - bbox_xyxy[0], bbox_xyxy[3] - bbox_xyxy[1]]
        anchor = torch.cat([torch.arange(20).unsqueeze(-1).repeat(1, 20).view(20, 20, 1) + 0.5,
                            torch.arange(20).unsqueeze(0).repeat(20, 1).view(20, 20, 1) + 0.5], dim=-1)
        lt, rb = torch.clip(anchor - torch.floor(bbox_patch[:2] - bbox_patch[2:] / 2), min=0), torch.clip(torch.ceil(bbox_patch[:2] + bbox_patch[2:] / 2) - anchor, min=0)
        center_mask = torch.sqrt((torch.min(lt[:, :, 0], rb[:, :, 0]) / torch.max(lt[:, :, 0], rb[:, :, 0])) *
                                 (torch.min(lt[:, :, 1], rb[:, :, 1]) / torch.max(lt[:, :, 1], rb[:, :, 1]))).T
        center_mask = center_mask * bbox_mask / center_mask.max()
        # center_mask = (center_mask * bbox_mask > 0)

        rle = cocomask.frPyObjects([seg], img.shape[1], img.shape[2])
        seg_mask = np.max(cocomask.decode(rle), axis=2).astype(np.float32)
        foreground_pixels = np.where(seg_mask == 1)
        centroid_x, centroid_y = round(np.mean(foreground_pixels[1])), round(np.mean(foreground_pixels[0]))
        mass_mask = torch.zeros(img.shape[1], img.shape[2])
        top, left, bottom, right = int(max(centroid_x - img.shape[2]/10, 0)), int(max(centroid_y - img.shape[1]/10, 0)),\
                                   int(min(centroid_x + img.shape[2]/10, img.shape[1])), int(min(centroid_y + img.shape[1]/10, img.shape[2]))
        mass_mask[left:right, top:bottom] = torch.Tensor(seg_mask[left:right, top:bottom] == 1)
        mass_mask = F.avg_pool2d(mass_mask.unsqueeze(0), img.shape[1] // 20).squeeze(0)

        return np.array(img, dtype=np.float32), np.array(img_mask, dtype=np.int16), \
               np.array(word_id, dtype=int), np.array(word_mask, dtype=np.int16), \
               np.array(bbox, dtype=np.float32), np.array(seg_mask, dtype=np.int8),\
               np.array(mass_mask, dtype=np.float32), np.array(center_mask, dtype=np.float32)

    def negative_sample(self, bbox):
        bbox_mask = torch.zeros([20, 20])
        bbox_mask[int(bbox[1]) - 2: int(bbox[1]) + 3, int(bbox[0]) - 2: int(bbox[0]) + 3] = 1
        flattened = bbox_mask.view(-1)
        ones_indices = (flattened == 0).nonzero(as_tuple=True)[0]
        random_index = torch.randperm(ones_indices.size(0))[0]
        random_position = ones_indices[random_index]
        row = random_position // bbox_mask.size(1)
        col = random_position % bbox_mask.size(1)
        negative_mask = torch.zeros([20, 20])
        negative_mask[row.item() - 1: row.item() + 2, col.item() - 1: col.item() + 2] = 1

        return negative_mask

