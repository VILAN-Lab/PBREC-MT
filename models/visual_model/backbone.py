# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from .resnet import *
from .fpn import FPN, LastLevelP6P7
from typing import Dict, List

from utils.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding
import loralib as lora


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        return [x for x in xs.values()]



class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 return_interm_layers: bool,
                 dilation: bool, train_cnn: bool):
        # TODO pretrained backbone
        backbone = eval(name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=False, norm_layer=FrozenBatchNorm2d)
            # pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        # lora.mark_only_lora_as_trainable(backbone)
        if not train_cnn:
            for p in backbone.parameters():
                p.requires_grad = False
        assert name in ('resnet50', 'resnet101')
        num_channels = 2048
        super().__init__(backbone, num_channels, return_interm_layers)


def conv_with_kaiming_uniform(use_gn=False, use_relu=False):
    def make_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            bias=False if use_gn else True
        )
        # Caffe2 implementation uses XavierFill, which in fact
        # corresponds to kaiming_uniform_ in PyTorch
        nn.init.kaiming_uniform_(conv.weight, a=1)
        if not use_gn:
            nn.init.constant_(conv.bias, 0)
        module = [conv,]
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv

    return make_conv


class Joiner(nn.Sequential):
    def __init__(self, backbone, fpn, position_embedding):
        super().__init__(backbone, fpn, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        ps = self[1](xs)
        out: List[NestedTensor] = []
        pos = []
        for p in ps:
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=p.shape[-2:]).to(torch.bool)[0]
            p = NestedTensor(p, mask)
            out.append(p)
            # position encoding
            pos.append(self[2](p).to(p.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    return_interm_layers = True
    train_cnn = args.lr_visu_cnn > 0
    backbone = Backbone(args.backbone, return_interm_layers, args.dilation, train_cnn)
    num_channels = 2048
    fpn = FPN(
        in_channels_list=[
            0,
            num_channels // 4,
            num_channels // 2,
            num_channels,
        ],
        out_channels=num_channels // 8,
        conv_block=conv_with_kaiming_uniform(),
        top_blocks=LastLevelP6P7(num_channels // 8, num_channels // 8),
    )
    model = Joiner(backbone, fpn, position_embedding)
    model.num_channels = backbone.num_channels
    return model
