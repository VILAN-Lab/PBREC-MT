#!/bin/bash

# # RefCOCO
#python -u -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --batch_size 32 --epochs 15 --aug_scale --aug_translate --dataset unc --max_query_len 30 --output_dir outputs/refcoco_seg --task seg

# # RefCOCO+
#python -u -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --batch_size 32 --epochs 15 --aug_scale --aug_translate --dataset unc+ --max_query_len 30 --output_dir outputs/refcocop_seg --task seg

# # RefCOCOg umd-split
python -u -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --batch_size 32 --epochs 15 --aug_scale --aug_translate --dataset gref_umd --max_query_len 40 --output_dir outputs/refcocog_seg --task boxseg
