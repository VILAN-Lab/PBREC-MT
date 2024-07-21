# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


"""Modulated Object Detection"""
class CA_module(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_ca_layers=2,
                 dim_feedforward=2048, dropout=0.1, activation="gelu",
                 return_intermediate_dec=True):
        super().__init__()
        ca_layer = AttentionLayer(d_model, nhead, dim_feedforward, dropout, activation)
        ca_norm = nn.LayerNorm(d_model)
        self.ca = AttentionModule(ca_layer, num_ca_layers, ca_norm,
                                  return_intermediate=return_intermediate_dec)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        self.norm = nn.LayerNorm(d_model)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, source, pos, mask):
        source = self.norm(source)
        if pos is not None:
            pos = pos.transpose(0, 1)
        emb = self.ca(source, pos, mask).transpose(1, 2)
        return emb


class AttentionModule(nn.Module):

    def __init__(self, attn_layer, num_layers, norm=None, return_intermediate=True):
        super().__init__()
        self.layers = _get_clones(attn_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, source,
                pos: Optional[Tensor] = None,
                mask: Optional[Tensor] = None):
        output = source

        intermediate = []

        for layer in self.layers:
            output = layer(output, pos, mask)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class AttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, source,
                pos: Optional[Tensor] = None,
                mask: Optional[Tensor] = None):
        q = k = self.with_pos_embed(source, pos)
        q2 = self.self_attn(q, k, value=source, key_padding_mask=mask)[0]
        q = q + self.dropout2(q2)
        q = self.norm2(q)
        q2 = self.linear2(self.dropout(self.activation(self.linear1(q))))
        q = q + self.dropout3(q2)
        q = self.norm3(q)
        return q


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_ca_layer(args, return_intermediate_dec=True):
    model = CA_module(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_ca_layers=args.ca_layers,
        dim_feedforward=args.dim_feedforward,
        activation=args.activation,
        return_intermediate_dec=return_intermediate_dec
    )
    return model


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _reset_parameters(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
