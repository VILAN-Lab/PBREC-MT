from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .visual_model.backbone import build_backbone
from .language_model.bert import build_bert
from .language_model.position_encoding import build_position_encoding
from .cross_attention import build_ca_layer
from utils.misc import NestedTensor


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.multi_scale = args.multi_scale
        self.task_type = args.task
        self.num_text_token = args.max_query_len
        self.backbone = build_backbone(args)
        self.bert = build_bert(args)
        self.txt_proj = nn.Linear(self.bert.num_channels, self.hidden_dim)
        self.position_embedding = build_position_encoding(self.hidden_dim, self.num_text_token)

        self.ca = build_ca_layer(args, return_intermediate_dec=True)


        if 'box' in self.task_type:
            self.box_mlp = MLP(self.hidden_dim, self.hidden_dim, 1, 3)
        else:
            self.box_mlp = None
        if 'seg' in self.task_type:
            self.seg_mlp = MLP(self.hidden_dim, self.hidden_dim, 1, 3)
        else:
            self.seg_mlp = None

    def forward(self, img_data, text_data):
        bs = img_data.tensors.shape[0]
        # visual backbone
        img_srcs, img_pos = self.backbone(img_data)
        img_src, img_mask, img_pos = \
            img_srcs[2].tensors.flatten(-2), img_srcs[2].mask.flatten(-2), img_pos[2].flatten(-2).transpose(1, 2)

        txt_fea = self.bert(text_data)
        txt_src, txt_mask = txt_fea.decompose()
        txt_src = self.txt_proj(txt_src)
        txt_pos = self.position_embedding(txt_src)
        assert txt_mask is not None

        # cross attn
        source = torch.cat([img_src.permute(2, 0, 1), txt_src.transpose(0, 1)], dim=0)
        pos = torch.cat([img_pos, txt_pos], dim=1)
        # pos = None
        mask = torch.cat([img_mask, txt_mask], dim=-1)
        vt_embs = self.ca(source, pos, mask)

        if self.box_mlp is not None:
            box_score = self.box_mlp(vt_embs[:, :, :400]).squeeze(-1)
        else:
            box_score = None
        if self.seg_mlp is not None:
            seg_score = self.seg_mlp(vt_embs[:, :, :400]).squeeze(-1)
        else:
            seg_score = None

        output = {'box_score': box_score,
                  'seg_score': seg_score,
                  'img_mask': img_mask}

        return output


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class random_erase(nn.Module):
    def __init__(self, drop_prob):
        super(random_erase, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x, y):
        assert x.dim() == y.dim()
        assert x.shape == y.shape
        if self.drop_prob < 0.0 or self.drop_prob > 1.0:
            raise ValueError(f"drop probability has to be between 0 and 1, but got {self.drop_prob}")
        if not self.training or self.drop_prob == 0.0:
            return (x + y) / 2
        layer, bs, _ = x.shape

        survival_rate = 1.0 - self.drop_prob
        size = [layer, bs, 1]
        noise_pos = torch.empty(size, dtype=x.dtype, device=x.device)
        noise_src = torch.empty(size, dtype=x.dtype, device=x.device)
        noise_pos = noise_pos.bernoulli_(survival_rate).to(torch.bool)
        noise_src = noise_src.bernoulli_(0.5).to(torch.bool)
        noise_x = noise_pos | noise_src
        noise_y = ~(noise_pos ^ noise_x)
        output = (x * noise_x + y * noise_y) / (noise_x.to(torch.int) + noise_y.to(torch.int))
        return output
