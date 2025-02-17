import os
import sys
import math
import copy
import logging
import requests
import zipfile
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig, OmegaConf

from src.data import RealDatasetCollection, SyntheticDatasetCollection
from src.data.cancer_sim.dataset import SyntheticCancerDataset
from src.models import TimeVaryingCausalModel
from src.models.utils import (
    grad_reverse,
    BRTreatmentOutcomeHead,
    AlphaRise,
    clip_normalize_stabilized_weights,
)
from src.models.utils_lstm import VariationalLSTM
from copy import deepcopy

logger = logging.getLogger(__name__)

# Data embedding before feeding into the model

# This will first one-hot encode all the categorical features and then embed them to n columns. The resulting columns will then be concatenated with the numerical features. The result will then be used to create the torch tensor for the model. The torch tensor will be shaped as (Cases, Time, Features).

# The input data will be a dataframe like this:


class DataEmbedder(nn.Module):
    def __init__(self, categorical_indices_sizes, numerical_indices, dataset):
        super(DataEmbedder, self).__init__()
        # dictionary with feature name, and a list of index and size
        self.categoricals = categorical_indices_sizes
        self.numerics = numerical_indices  # dictionary with feature name and index
        self.embeddings = nn.ModuleDict()
        self.mapping_dicts = {}

        # Initialize embeddings and mapping dictionaries
        for key in self.categoricals:
            unique_values = np.unique(dataset[:, :, self.categoricals[key][0]])
            self.mapping_dicts[key] = {
                name: idx for idx, name in enumerate(unique_values)
            }
            self.embeddings[key] = nn.Embedding(
                num_embeddings=len(unique_values),
                embedding_dim=self.categoricals[key][1],
            )
            print(
                f"Feature: {key}, Categories: {len(unique_values)}, Embedding Size: {self.categoricals[key][1]}"
            )

    def forward(self, dataset):
        # Apply embeddings to the categorical indices
        if len(self.categoricals) == 0:
            return dataset
        else:
            embedded_features = []
            for key in self.categoricals:
                # Map the categorical values to their corresponding indices
                indices = dataset[:, :,
                                  self.categoricals[key][0]].cpu().numpy()
                mapped_indices = np.vectorize(
                    self.mapping_dicts[key].get)(indices)
                mapped_indices = torch.tensor(
                    mapped_indices, dtype=torch.long, device=dataset.device
                )
                # print(f"Feature: {key}, Mapped Indices: {mapped_indices}")
                embedded_features.append(self.embeddings[key](mapped_indices))

            embedded_features = torch.cat(embedded_features, dim=-1)

            numeric_features = dataset[:, :, list(
                self.numerics.values())].float()

            # Concatenate the embedded features with the numerical data
            result = torch.cat([embedded_features, numeric_features], dim=-1)

            feature_count_embedded = len(self.numerics) + sum(
                [self.categoricals[key][1] for key in self.categoricals]
            )

            result = result.reshape(
                dataset.shape[0], -1, feature_count_embedded)

            return result

class moded_TimesSeriesAttention(nn.Module):
    """
    A module that computes multi-head attention given query, key, and value tensors for time series data of shape (b, t, f, e)
    """

    def __init__(self, embed_dim: int, num_heads: int):
        """
        Constructor.

        Inputs:
        - input_dim: Dimension of the input query, key, and value. We assume they all have
          the same dimensions. This is basically the dimension of the embedding.
        - num_heads: Number of attention heads
        """
        super(moded_TimesSeriesAttention, self).__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_per_head = embed_dim // num_heads

        self.linear_query = nn.Linear(embed_dim, embed_dim)
        self.linear_key = nn.Linear(embed_dim, embed_dim)
        self.linear_value = nn.Linear(
            embed_dim, embed_dim
        )  # (self.num_heads * self.dim_per_head * self.dim_per_head))
        self.output_linear = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax2d()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """
        Compute the attended feature representations.

        Inputs:
        - query: Tensor of the shape BxTxFXE, where B is the batch size, T is the time dimension, F is the feature dimension,
        and E is the embedding dimension
        - key: Tensor of the shape BxTxFXE
        - value: Tensor of the shape BxTxFXE
        - mask: Tensor indicating where the attention should *not* be performed
        """
        b = query.shape[0]
        t = query.shape[1]
        f = query.shape[2]
        e = query.shape[3]
        d = self.dim_per_head
        h = self.num_heads

        query_linear = self.linear_query(query)
        key_linear = self.linear_key(key)
        value_linear = self.linear_value(value)

        query_reshaped = query_linear.reshape(
            b, t, f, self.num_heads, self.dim_per_head
        )
        key_reshaped = key_linear.reshape(
            b, t, f, self.num_heads, self.dim_per_head)
        value_reshaped = value_linear.reshape(
            b, t, f, self.num_heads, self.dim_per_head
        )  # , self.dim_per_head)

        query_reshaped = query_reshaped.permute(0, 3, 1, 2, 4)  # BxHxTxFxD
        key_reshaped = key_reshaped.permute(0, 3, 1, 2, 4)  # BxHxTxFxD
        value_reshaped = value_reshaped.permute(
            0, 3, 1, 2, 4)  # , 5) # BxHxTxFxDxD

        kq = torch.einsum("bhtfd,bhxyd->bhtfxy", key_reshaped, query_reshaped)

        dot_prod_scores = kq / math.sqrt(self.dim_per_head)

        # softmax across last 2 features (use softmax2d)
        dot_prod_scores = dot_prod_scores.reshape(b * h, t * f, t, f)
        dot_prod_scores = self.softmax(dot_prod_scores)
        dot_prod_scores = dot_prod_scores.reshape(b, h, t, f, t, f)

        out = torch.einsum("bhtfxy,bhtfd->bhtfd",
                           dot_prod_scores, value_reshaped)
        out = out.permute(0, 2, 3, 1, 4).reshape(b, t, f, e)
        out = self.output_linear(out)

        return out
    
class moded_TransformerEncoderCell(nn.Module):
    """
    A single cell (unit) for the Transformer encoder.
    """

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float):
        """
        Inputs:
        - embed_dim: embedding dimension for each element in the time series data
        - num_heads: Number of attention heads in a multi-head attention module
        - ff_dim: The hidden dimension for a feedforward network
        - dropout: Dropout ratio for the output of the multi-head attention and feedforward
          modules.
        """
        super(moded_TransformerEncoderCell, self).__init__()

        self.time_series_attention = moded_TimesSeriesAttention(
            embed_dim, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Inputs:
        - x: Tensor of the shape BxTxFXE, where B is the batch size, T is the time dimension, F is the feature dimension,
        and E is the embedding dimension
        - mask: Tensor for multi-head attention
        """

        attention2 = self.time_series_attention(x, x, x, mask)
        attention = x + self.dropout1(attention2)
        attention = self.layer_norm(attention)

        attention2 = self.linear2(
            self.dropout(self.activation(self.linear1(attention)))
        )
        attention = attention + self.dropout2(attention2)
        attention = self.layer_norm(attention)

        return attention
class moded_TransformerEncoder(nn.Module):
    """
    A full encoder consisting of a set of TransformerEncoderCell.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        num_cells: int,
        dropout: float = 0.1,
    ):
        """
        Inputs:
        - embed_dim: embedding dimension for each element in the time series data
        - num_heads: Number of attention heads in a multi-head attention module
        - ff_dim: The hidden dimension for a feedforward network
        - num_cells: Number of time series attention cells in the encoder
        - dropout: Dropout ratio for the output of the multi-head attention and feedforward
          modules.
        """
        super(moded_TransformerEncoder, self).__init__()

        self.norm = None

        self.encoder_modules = nn.ModuleList(
            moded_TransformerEncoderCell(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_cells)
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Inputs:
        - x: Tensor of the shape BxTxFXE, where B is the batch size, T is the time dimension, F is the feature dimension,
        and E is the embedding dimension
        - mask: Tensor for multi-head attention

        Return:
        - y: Tensor of the shape BxTxFXE
        """

        # run encoder modules and add residual connections
        for encoder_module in self.encoder_modules:
            x = encoder_module(x, mask)

        y = x

        return y

## RSA
class TimesSeriesAttention(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        nh=8,
        dk=0,
        dv=0,
        dd=0,
        kernel_size=(3, 7),
        stride=(1, 1, 1),
        kernel_type="VplusR",  # ['V', 'R', 'VplusR']
        feat_type="VplusR",  # ['V', 'R', 'VplusR']
    ):
        super(TimesSeriesAttention, self).__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.nh = nh
        self.dv = dv = d_out // nh if dv == 0 else dv
        self.dk = dk = dv if dk == 0 else dk
        self.dd = dd = dk if dd == 0 else dd

        self.kernel_size = kernel_size
        self.stride = stride
        self.kernel_type = kernel_type
        self.feat_type = feat_type

        assert self.kernel_type in [
            "V",
            "R",
            "VplusR",
        ], "Not implemented involution type: {}".format(self.kernel_type)
        assert self.feat_type in [
            "V",
            "R",
            "VplusR",
        ], "Not implemented feature type: {}".format(self.feat_type)

        # print("d_in: {}, d_out: {}, nh: {}, dk: {}, dv: {}, dd:{}, kernel_size: {}, kernel_type: {}, feat_type: {}"
        #       .format(d_in, d_out, nh, dk, dv, self.dd, kernel_size, kernel_type, feat_type))

        self.ksize = ksize = kernel_size[0] * kernel_size[1]
        self.pad = pad = tuple(k // 2 for k in kernel_size)

        # hidden dimension
        d_hid = nh * dk + dv if self.kernel_type == "V" else nh * dk + dk + dv

        # Linear projection
        # self.projection = nn.Conv2d(d_in, d_hid, 1, bias=False)
        self.projection_linear = nn.Sequential(
            nn.Linear(d_in, d_hid, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(d_hid, d_hid, bias=False),
        )

        # Intervolution Kernel
        if self.kernel_type == "V":
            self.H2 = nn.Conv2d(1, dd, kernel_size,
                                padding=self.pad, bias=False)
        elif self.kernel_type == "R":
            self.H1 = nn.Conv2d(
                dk, dk * dd, kernel_size, padding=self.pad, groups=dk, bias=False
            )
            self.H2 = nn.Conv2d(1, dd, kernel_size,
                                padding=self.pad, bias=False)
        elif self.kernel_type == "VplusR":
            self.P1 = nn.Parameter(
                torch.randn(dk, dd).unsqueeze(0) * np.sqrt(1 / (ksize * dd)),
                requires_grad=True,
            )
            self.H1 = nn.Conv2d(
                dk, dk * dd, kernel_size, padding=self.pad, groups=dk, bias=False
            )
            self.H2 = nn.Conv2d(1, dd, kernel_size,
                                padding=self.pad, bias=False)
        else:
            raise NotImplementedError

        # Feature embedding layer
        if self.feat_type == "V":
            pass
        elif self.feat_type == "R":
            self.G = nn.Conv2d(1, dv, kernel_size,
                               padding=self.pad, bias=False)
        elif self.feat_type == "VplusR":
            self.G = nn.Conv2d(1, dv, kernel_size,
                               padding=self.pad, bias=False)
            self.I = nn.Parameter(
                torch.eye(dk).unsqueeze(0), requires_grad=True)
        else:
            raise NotImplementedError

        # Downsampling layer
        if max(self.stride) > 1:
            self.avgpool = nn.AvgPool2d(
                kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)
            )

    def L2norm(self, x, d=1):
        eps = 1e-6
        norm = x**2
        norm = norm.sum(dim=d, keepdim=True) + eps
        norm = norm ** (0.5)
        return x / norm

    def forward(self, x):

        # print(x.shape)
        x = x.permute(0, 3, 1, 2)
        N, C, T, H = x.shape

        x = x.permute(0, 2, 3, 1)

        """Linear projection"""
        # x_proj = self.projection(x)
        x_proj = self.projection_linear(x)
        x_proj = x_proj.permute(0, 3, 1, 2)
        # print(x_proj.shape)

        if self.kernel_type != "V":
            q, k, v = torch.split(
                x_proj, [self.nh * self.dk, self.dk, self.dv], dim=1)
        else:
            q, v = torch.split(x_proj, [self.nh * self.dk, self.dv], dim=1)

        """Normalization"""
        q = rearrange(q, "b (nh k) t h -> b nh k t h", k=self.dk)
        q = self.L2norm(q, d=2)
        q = rearrange(q, "b nh k t h -> (b t h) nh k")

        v = self.L2norm(v, d=1)

        if self.kernel_type != "V":
            k = self.L2norm(k, d=1)

        """
        q = (b t h) nh k
        k = b k t h
        v = b v t h
        """

        # Intervolution generation
        # Basic kernel
        if self.kernel_type == "V":
            kernel = q
        # Relational kernel
        else:
            K_H1 = self.H1(k)
            K_H1 = rearrange(K_H1, "b (k d) t h-> (b t h) k d", k=self.dk)

            if self.kernel_type == "VplusR":
                K_H1 = K_H1 + self.P1

            kernel = torch.einsum(
                "abc,abd->acd", q.transpose(1, 2), K_H1
            )  # (bth, nh, d)

        # feature generation
        # Appearance feature
        v = rearrange(v, "b (v 1) t h-> (b v) 1 t h")

        V = self.H2(v)  # (bv, d, t, h)
        feature = rearrange(V, "(b v) d t h -> (b t h) v d", v=self.dv)

        # Relational feature
        if self.feat_type in ["R", "VplusR"]:
            V_G = self.G(v)  # (bv, v2, t, h)
            V_G = rearrange(V_G, "(b v) v2 t h -> (b t h) v v2", v=self.dv)

            if self.feat_type == "VplusR":
                V_G = V_G + self.I

            feature = torch.einsum("abc,abd->acd", V_G,
                                   feature)  # (bth, v2, d)

        # kernel * feat
        out = torch.einsum("abc,adc->adb", kernel, feature)  # (bth, nh, v2)

        out = rearrange(out, "(b t h) nh v -> b (nh v) t h", t=T, h=H)

        if max(self.stride) > 1:
            out = self.avgpool(out)

        out = out.permute(0, 2, 3, 1)

        return out
class TransformerEncoderCell(nn.Module):
    """
    A single cell (unit) for the Transformer encoder.
    """

    def __init__(
        self, embed_dim: int, num_heads: int, kernel_size, ff_dim: int, dropout: float
    ):
        """
        Inputs:
        - embed_dim: embedding dimension for each element in the time series data
        - num_heads: Number of attention heads in a multi-head attention module
        - ff_dim: The hidden dimension for a feedforward network
        - dropout: Dropout ratio for the output of the multi-head attention and feedforward
          modules.
        """
        super(TransformerEncoderCell, self).__init__()

        self.time_series_attention = TimesSeriesAttention(
            embed_dim, embed_dim, nh=num_heads, kernel_size=kernel_size
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, data: torch.Tensor, embeddings, mask: torch.Tensor = None):
        """
        Inputs:
        - x: Tensor of the shape BxTxFXE, where B is the batch size, T is the time dimension, F is the feature dimension,
        and E is the embedding dimension
        - mask: Tensor for multi-head attention
        """

        # attention2 = self.time_series_attention(x, x, x, mask)
        attention2 = self.time_series_attention(data)
        attention = data + self.dropout1(attention2)
        attention = self.layer_norm(attention)

        attention2 = self.linear2(
            self.dropout(self.activation(self.linear1(attention)))
        )
        attention = attention + self.dropout2(attention2)
        attention = self.layer_norm(attention)

        return attention
class TransformerEncoder(nn.Module):
    """
    A full encoder consisting of a set of TransformerEncoderCell.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kernel_size,
        ff_dim: int,
        num_cells: int,
        dropout: float = 0.1,
    ):
        """
        Inputs:
        - embed_dim: embedding dimension for each element in the time series data
        - num_heads: Number of attention heads in a multi-head attention module
        - ff_dim: The hidden dimension for a feedforward network
        - num_cells: Number of time series attention cells in the encoder
        - dropout: Dropout ratio for the output of the multi-head attention and feedforward
          modules.
        """
        super(TransformerEncoder, self).__init__()

        self.norm = None

        self.encoder_modules = nn.ModuleList(
            TransformerEncoderCell(embed_dim, num_heads,
                                   kernel_size, ff_dim, dropout)
            for _ in range(num_cells)
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Inputs:
        - x: Tensor of the shape BxTxFXE, where B is the batch size, T is the time dimension, F is the feature dimension,
        and E is the embedding dimension
        - mask: Tensor for multi-head attention

        Return:
        - y: Tensor of the shape BxTxFXE
        """

        # run encoder modules and add residual connections
        for encoder_module in self.encoder_modules:
            x = encoder_module(x, mask)

        y = x

        return y
## CSDI transformer

def get_torch_trans(num_heads=8, num_cells=1, embed_dim=128, ff_dim=512, dropout=0.1):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=ff_dim,
        activation="gelu",
        dropout=dropout,
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=num_cells)
## Embeddings

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim, projection_dim=None):
        super(DiffusionEmbedding, self).__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, embedding_dim)

    def forward(self, diffusion_step, data, device="cpu"):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        x = torch.zeros(data.shape).to(device) + x.unsqueeze(1).unsqueeze(1)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(
            0
        )  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat(
            [torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_len=10000.0):
        super(TimeEmbedding, self).__init__()
        self.max_len = max_len
        self.learnable = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, data, device="cpu"):

        b, l, f, e = data.shape
        pe = None
        pe_row = torch.arange(l)

        pe = pe_row.unsqueeze(0)
        pe = pe.unsqueeze(2)

        pe = pe.repeat((b, 1, e))
        pe = pe.float()

        pe[:, :, 0::2] = torch.sin(
            pe[:, :, 0::2] / (self.max_len ** (torch.arange(0, e, 2) / e))
        )
        pe[:, :, 1::2] = torch.cos(
            pe[:, :, 1::2] / (self.max_len ** (torch.arange(0, e, 2) / e))
        )

        pe = pe.to(device).unsqueeze(2).repeat((1, 1, f, 1))

        # pe = torch.arange(l).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)
        # pe = torch.zeros(data.shape).to(device) + pe

        # div_term = 1 / torch.pow(
        #     self.max_len, torch.arange(0, f, 2) / f
        # ).unsqueeze(-1).to(device)

        # pe[:, :, 0::2] = torch.sin(pe[:, :, 0::2] * div_term)
        # pe[:, :, 1::2] = torch.cos(pe[:, :, 1::2] * div_term)

        return self.learnable(pe)


class FeatureEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_len=10000.0):
        super(FeatureEmbedding, self).__init__()
        self.max_len = max_len
        self.learnable = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, data, device="cpu"):
        b, l, f, e = data.shape

        pe = None
        pe_row = torch.arange(f)

        pe = pe_row.unsqueeze(0)
        pe = pe.unsqueeze(2)

        pe = pe.repeat((b, 1, e))
        pe = pe.float()

        pe[:, :, 0::2] = torch.sin(
            pe[:, :, 0::2] / (self.max_len ** (torch.arange(0, e, 2) / e))
        )
        pe[:, :, 1::2] = torch.cos(
            pe[:, :, 1::2] / (self.max_len ** (torch.arange(0, e, 2) / e))
        )

        pe = pe.to(device).unsqueeze(1).repeat((1, l, 1, 1))

        # pe = torch.arange(f).unsqueeze(0).unsqueeze(0).unsqueeze(-1).to(device)
        # pe = torch.zeros(data.shape).to(device) + pe

        # div_term = 1 / torch.pow(
        #     self.max_len, torch.arange(0, e, 2) / e
        # ).to(device)

        # pe[:, :, :, 0::2] = torch.sin(pe[:, :, :, 0::2] * div_term)
        # pe[:, :, :, 1::2] = torch.cos(pe[:, :, :, 1::2] * div_term)

        return self.learnable(pe)
# Residual block

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer
class ResidualBlock(nn.Module):
    def __init__(
        self,
        num_heads=8,
        num_cells=1,
        kernel_size=(3, 7),
        embed_dim=128,
        ff_dim=512,
        dropout=0.1,
        method="rsa",
    ):
        super().__init__()

        self.method = method

        self.embedding_add = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self.layer_norm = nn.LayerNorm(embed_dim)

        self.mid_projection = Conv1d_with_init(embed_dim, 2 * embed_dim, 1)
        # nn.Linear(embed_dim, embed_dim*2)
        self.output_projection = Conv1d_with_init(embed_dim, 2 * embed_dim, 1)
        # self.output_projection = nn.Linear(embed_dim, embed_dim*2)

        if method == "rsa":
            self.feature_and_time_transformer = TransformerEncoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                kernel_size=kernel_size,
                ff_dim=ff_dim,
                num_cells=num_cells,
                dropout=dropout,
            )
            self.linear_time_and_feature = nn.Linear(embed_dim, embed_dim)

        elif method == "csdi":
            self.time_layer = get_torch_trans(
                num_heads=num_heads,
                num_cells=num_cells,
                embed_dim=embed_dim,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            self.feature_layer = get_torch_trans(
                num_heads=num_heads,
                num_cells=num_cells,
                embed_dim=embed_dim,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            self.linear_time = nn.Linear(embed_dim, embed_dim)
            self.linear_feature = nn.Linear(embed_dim, embed_dim)

        elif method == "csdi_moded_transformer":
            self.time_layer = get_torch_trans(
                num_heads=num_heads,
                num_cells=num_cells,
                embed_dim=embed_dim,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            self.feature_layer = get_torch_trans(
                num_heads=num_heads,
                num_cells=num_cells,
                embed_dim=embed_dim,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            self.linear_time = nn.Linear(embed_dim, embed_dim)
            self.linear_feature = nn.Linear(embed_dim, embed_dim)
            self.feature_and_time_transformer = moded_TransformerEncoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_cells=num_cells,
                dropout=dropout,
            )
            self.linear_time_and_feature = nn.Linear(embed_dim, embed_dim)

        elif method == "rsa_csdi":
            self.time_layer = get_torch_trans(
                num_heads=num_heads,
                num_cells=num_cells,
                embed_dim=embed_dim,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            self.feature_layer = get_torch_trans(
                num_heads=num_heads,
                num_cells=num_cells,
                embed_dim=embed_dim,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            self.linear_time = nn.Linear(embed_dim, embed_dim)
            self.linear_feature = nn.Linear(embed_dim, embed_dim)
            self.feature_and_time_transformer = TransformerEncoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                kernel_size=kernel_size,
                ff_dim=ff_dim,
                num_cells=num_cells,
                dropout=dropout,
            )
            self.linear_time_and_feature = nn.Linear(embed_dim, embed_dim)

        elif method == "rsa_moded_transformer":
            self.feature_and_time_transformer = TransformerEncoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                kernel_size=kernel_size,
                ff_dim=ff_dim,
                num_cells=num_cells,
                dropout=dropout,
            )
            self.linear_time_and_feature = nn.Linear(embed_dim, embed_dim)
            self.moded_feature_and_time_transformer = moded_TransformerEncoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_cells=num_cells,
                dropout=dropout,
            )
            self.moded_linear_time_and_feature = nn.Linear(
                embed_dim, embed_dim)

        elif method == "moded_transformer_alone":
            self.moded_feature_and_time_transformer = moded_TransformerEncoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_cells=num_cells,
                dropout=dropout,
            )
            self.moded_linear_time_and_feature = nn.Linear(
                embed_dim, embed_dim)

        else:
            raise NotImplementedError

        logging.info("Initializing ResidualBlock with method: %s", method)

    def forward_time(self, y, base_shape):
        b, t, f, e = base_shape
        y = y.permute(0, 2, 1, 3).reshape(b * f, t, e)
        y = self.time_layer(y.permute(1, 0, 2)).permute(1, 0, 2)
        y = y.reshape(b, f, t, e).permute(0, 2, 1, 3)
        return y

    def forward_feature(self, y, base_shape):
        b, t, f, e = base_shape
        y = y.reshape(b * t, f, e)
        y = self.feature_layer(y.permute(1, 0, 2)).permute(1, 0, 2)
        y = y.reshape(b, t, f, e)
        return y

    def forward(self, noised_data, diffusion_emb, time_emb, feature_emb):

        logging.info("ResidualBlock forward started")

        b, t, f, e = noised_data.shape
        base_shape = noised_data.shape

        y = torch.stack((noised_data, diffusion_emb,
                        time_emb, feature_emb), dim=-1)
        y = y.reshape(b, t, f, -1)
        y = self.embedding_add(y)
        y_resid = y

        if self.method == "rsa":
            y = self.feature_and_time_transformer(y)
            y = y.squeeze(-1)
            y = self.linear_time_and_feature(y)

        elif self.method == "csdi":
            y = self.forward_time(y, base_shape)
            y = self.linear_time(y)
            y = (y + y_resid) / math.sqrt(2.0)
            y = self.layer_norm(y)
            y = self.forward_feature(y, base_shape)
            y = self.linear_feature(y)

        elif self.method == "csdi_moded_transformer":
            y = self.forward_time(y, base_shape)
            y = self.linear_time(y)
            y = (y + y_resid) / math.sqrt(2.0)
            y = self.layer_norm(y)
            y = self.forward_feature(y, base_shape)
            y = self.linear_feature(y)
            y = (y + y_resid) / math.sqrt(2.0)
            y_resid = y
            y = self.layer_norm(y)
            y = self.feature_and_time_transformer(y)
            y = self.linear_time_and_feature(y)

        elif self.method == "rsa_csdi":
            y = self.forward_time(y, base_shape)
            y = self.linear_time(y)
            y = (y + y_resid) / math.sqrt(2.0)
            y = self.layer_norm(y)
            y = self.forward_feature(y, base_shape)
            y = self.linear_feature(y)
            y = (y + y_resid) / math.sqrt(2.0)
            y_resid = y
            y = self.layer_norm(y)
            y = self.feature_and_time_transformer(y)
            y = y.squeeze(-1)
            y = self.linear_time_and_feature(y)

        elif self.method == "rsa_moded_transformer":
            y = self.feature_and_time_transformer(y)
            y = y.squeeze(-1)
            y = self.linear_time_and_feature(y)
            y = (y + y_resid) / math.sqrt(2.0)
            y = self.layer_norm(y)
            y = self.moded_feature_and_time_transformer(y)
            y = self.moded_linear_time_and_feature(y)

        elif self.method == "moded_transformer_alone":
            y = self.moded_feature_and_time_transformer(y)
            y = self.moded_linear_time_and_feature(y)

        y = (y + y_resid) / math.sqrt(2.0)
        y = self.layer_norm(y)
        y = y.permute(0, 3, 1, 2).reshape(b, e, t * f)
        y = self.mid_projection(y)
        # y = y.permute(0, 3, 2, 1).reshape(b, 2*e, t*f)

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (b,e,f*t)
        # y = y.permute(0, 2, 1)
        y = self.output_projection(y)
        # y = y.permute(0, 2, 1)

        residual, skip = torch.chunk(y, 2, dim=1)
        residual = residual.permute(0, 2, 1)
        skip = skip.permute(0, 2, 1)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)

        logging.info("ResidualBlock forward completed")

        return (noised_data + residual) / math.sqrt(2.0), skip
class ModelLoop(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        diffusion_steps=1000,
        num_heads=8,
        kernel_size=(3, 7),
        num_cells=1,
        num_residual_layers=4,
        ff_dim=512,
        dropout=0.1,
        method="rsa",
        device="cpu",
    ):
        super().__init__()

        self.device = device
        self.emb_dim = embed_dim

        # self.data_embedding_linear = nn.Sequential(
        #     nn.Linear(1, self.emb_dim),
        #     nn.SiLU(),
        #     nn.Linear(self.emb_dim, self.emb_dim)
        # )
        # self.x_embedding = nn.Sequential(
        #     nn.Linear(1, self.emb_dim),
        #     nn.SiLU(),
        #     nn.Linear(self.emb_dim, self.emb_dim)
        # )

        self.data_embedding_linear = Conv1d_with_init(1, self.emb_dim, 1)
        self.x_embedding = Conv1d_with_init(2, self.emb_dim, 1)

        self.output = Conv1d_with_init(self.emb_dim, 1, 1)
        self.output_final = Conv1d_with_init(self.emb_dim, 1, 1)

        # self.x_add = nn.Sequential(
        #     nn.Linear(embed_dim*num_residual_layers, embed_dim*num_residual_layers),
        #     nn.SiLU(),
        #     nn.Linear(embed_dim*num_residual_layers, embed_dim)
        # )

        self.diffusion_embedding = DiffusionEmbedding(
            diffusion_steps, embed_dim)
        self.time_embedding = TimeEmbedding(embed_dim)
        self.feature_embedding = FeatureEmbedding(embed_dim)

        self.residual_layers = nn.ModuleList(
            ResidualBlock(
                num_heads=num_heads,
                num_cells=num_cells,
                kernel_size=kernel_size,
                embed_dim=embed_dim,
                ff_dim=ff_dim,
                dropout=dropout,
                method=method,
            )
            for _ in range(num_residual_layers)
        )

        # self.output = nn.Sequential(
        #     nn.Linear(self.emb_dim, self.emb_dim),
        #     nn.SiLU(),
        #     nn.Linear(self.emb_dim, 1)
        # )

        # self.output_final = nn.Sequential(
        #     nn.Linear(self.emb_dim, self.emb_dim),
        #     nn.SiLU(),
        #     nn.Linear(self.emb_dim, 1)
        # )

        logging.info("Initializing ModelLoop with embed_dim: %s, method: %s", embed_dim, method)

    def forward(self, noised_data, noise_mask, diffusion_t):

        logging.info("ModelLoop forward: noised_data shape: %s", noised_data.shape)

        b, t, f, a = noised_data.shape

        noised_data_reshaped = noised_data.permute(
            0, 3, 1, 2).reshape(b, 1, t * f)
        noised_data_embedded = (
            self.data_embedding_linear(noised_data_reshaped)
            .permute(0, 2, 1)
            .reshape(b, t, f, self.emb_dim)
        )
        diffusion_embedding = self.diffusion_embedding(
            diffusion_t, noised_data_embedded, device=self.device
        )
        time_embedding = self.time_embedding(
            noised_data_embedded, device=self.device)
        feature_embedding = self.feature_embedding(
            noised_data_embedded, device=self.device
        )

        x = noised_data_embedded
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(
                x, diffusion_embedding, time_embedding, feature_embedding
            )
            skip.append(skip_connection)
            x = x.permute(0, 3, 1, 2).reshape(b, self.emb_dim, t * f)
            x = self.output(x).permute(0, 2, 1).reshape(b, t, f)
            x = torch.stack((x, noised_data.squeeze(-1)), dim=-1)
            # x = x * noise_mask + noised_data * (1 - noise_mask)
            x = x.permute(0, 3, 1, 2).reshape(b, 2, t * f)
            x = self.x_embedding(x).permute(
                0, 2, 1).reshape(b, t, f, self.emb_dim)

        x = torch.sum(torch.stack(skip, dim=-1), dim=-1) / math.sqrt(
            len(self.residual_layers)
        )
        # x = torch.stack(skip, dim = -1).reshape(b, t, f, -1)
        # x = self.x_add(x)
        x = x.permute(0, 3, 1, 2).reshape(b, self.emb_dim, t * f)
        x = self.output_final(x).permute(
            0, 2, 1).reshape(b, t, f, 1).squeeze(-1)

        logging.info("ModelLoop forward: output shape: %s", x.shape)

        return x
# Beta Schedules

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, num_diffusion_timesteps)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == "quadratic":
        scale = 50 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.5
        return (
            torch.linspace(beta_start**0.5, beta_end**0.5,
                           num_diffusion_timesteps) ** 2
        )

    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)
# Imputer

class diffusion_imputation(nn.Module):
    def __init__(
        self,
        emb_dim,
        excluded_features=None,
        # vocab_size,
        # pad_idx= None,
        strategy="random",
        num_residual_layers=4,
        features_to_impute=None,
        features_to_impute_completely=None,
        features_to_impute_after_time=None,
        last_n_time=1,
        missing_prp=0.1,
        diffusion_steps=1000,
        diffusion_beta_schedule="quadratic",
        num_heads=8,
        kernel_size=(3, 7),
        ff_dim=512,
        num_cells=2,
        dropout=0.1,
        method="rsa",
        device="cpu",
        sequence_length=None
    ):

        super().__init__()

        self.device = device
        self.emb_dim = emb_dim
        self.strategy = strategy
        self.features_to_impute = features_to_impute
        self.missing_prp = missing_prp
        self.diffusion_steps = diffusion_steps
        self.last_n_time = last_n_time
        self.exclude_features = excluded_features
        self.features_to_impute_completely = features_to_impute_completely
        self.features_to_impute_after_time = features_to_impute_after_time
        self.sequence_length = sequence_length

        # set device to cuda if available
        if torch.cuda.is_available():
            self.device = "cuda"

        self.model_loop = ModelLoop(
            embed_dim=self.emb_dim,
            diffusion_steps=diffusion_steps,
            num_heads=num_heads,
            kernel_size=kernel_size,
            ff_dim=ff_dim,
            num_cells=num_cells,
            dropout=dropout,
            num_residual_layers=num_residual_layers,
            method=method,
            device=self.device,
        )

        self.beta = get_named_beta_schedule(
            diffusion_beta_schedule, diffusion_steps)

        # self.beta = torch.linspace(0.0001, 0.5, diffusion_steps)

        # self.beta = torch.linspace(
        #         0.0001 ** 0.5, 0.5 ** 0.5, diffusion_steps
        #     ) ** 2

        self.alpha_hat = 1 - self.beta
        self.alpha = torch.cumprod(self.alpha_hat, dim=0)
        self.alpha_torch = torch.tensor(self.alpha).float()

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_mask(self, data, strategy="random"):

        b = data.shape[0]
        t = data.shape[1]
        f = data.shape[2]

        if strategy == "forecasting":
            forecasted_time = torch.randint(2, t, (b, 1, 1))
            mask = torch.zeros_like(data)
            for i in range(b):
                mask[i, forecasted_time[i]:, :] = 1

        if strategy == "forecasting_last_n_time":
            mask = torch.zeros_like(data)
            mask[:, -self.last_n_time, :] = 1

        if strategy == "death_prediction":
            mask = torch.zeros_like(data)
            # death is the last 7 columns of the data
            mask[:, :, -1] = 1

        if strategy == "random_features":
            selected_features = torch.randint(0, f, (b, 1, 1))
            mask = torch.zeros_like(data)
            mask[:, :, selected_features] = 1

        if strategy == "selected_features":
            mask = torch.zeros_like(data)
            mask[:, :, self.features_to_impute] = 1

        if strategy == "selected_features_after_time":
            selected_time = torch.randint(1, t, (b, 1, 1))
            mask = torch.zeros_like(data)
            mask[:, selected_time:, self.features_to_impute] = 1

        if strategy == "selected_features_last_n_time":
            mask = torch.zeros_like(data)
            mask[:, -self.last_n_time:, self.features_to_impute] = 1

        if strategy == "selected_features_last_n_sequence_length":
            assert self.sequence_length is not None
            mask = torch.zeros_like(data)
            for i in range(self.sequence_length.shape[0]):
                sequence_length = int(self.sequence_length[i])
                if i < mask.shape[0]:
                    mask[i, (sequence_length - self.last_n_time)
                             :sequence_length, self.features_to_impute] = 1

        if strategy == "whole_sequence":
            mask = torch.ones_like(data)

        if strategy == "random":
            mask = torch.rand(size=(b, t, f))
            mask = mask < self.missing_prp
            mask = mask.float()

        if strategy == "selected_features_and_selected_features_after_time":
            mask = torch.zeros_like(data)
            mask[:, :, self.features_to_impute_completely] = 1
            mask[:, -self.last_n_time:, self.features_to_impute_after_time] = 1

        if self.exclude_features is not None:
            mask[:, :, self.exclude_features] = 0

        return mask

    def loss_func(self, predicted_noise, noise, noise_mask):
        # noise = torch.nan_to_num(noise, nan=0.0)
        # predicted_noise = torch.nan_to_num(predicted_noise, nan=0.0)
        residual = noise - predicted_noise
        num_obs = torch.sum(noise_mask)
        loss = (residual**2).sum() / num_obs
        return loss

    def weighted_loss_func(self, predicted_noise, noise, noise_mask, stabilized_weights):
        # Calculate the residuals
        residual = noise - predicted_noise

        # Get the sample weights
        # print(f"stabilized_weights shape: {stabilized_weights.shape}")
        sw = stabilized_weights.to(self.device)
        # clip sw at 5th and 95th percentile
        sw = torch.clamp(sw, 0.05, 0.95)
        sw = sw.unsqueeze(-1).repeat(1, 1, residual.shape[-1]) * noise_mask
        # print(residual.shape)
        # print(sw)
        # Apply the sample weights to the squared residuals
        weighted_squared_residuals = (residual**2) * sw

        # Sum the weighted squared residuals
        weighted_loss_sum = weighted_squared_residuals.sum()

        # Normalize the loss by the sum of the weights
        loss = weighted_loss_sum / sw.sum()

        # print(loss)
        return loss

    def explode_trajectories(self, data, projection_horizon):

        self.data = data
        # assert self.processed

        # logger.info(f'Exploding {self.subset_name} dataset before testing (multiple sequences)')

        outputs = self.data['outputs']
        prev_outputs = self.data['prev_outputs']
        sequence_lengths = self.data['sequence_lengths']
        # vitals = self.data['vitals']
        # next_vitals = self.data['next_vitals']
        active_entries = self.data['active_entries']
        current_treatments = self.data['current_treatments']
        previous_treatments = self.data['prev_treatments']
        static_features = self.data['static_features']
        # repeat static features t times (first dimension in outputs)
        static_features = static_features.unsqueeze(
            1).repeat(1, outputs.shape[1], 1)
        if 'stabilized_weights' in self.data:
            stabilized_weights = self.data['stabilized_weights']

        num_patients, max_seq_length, num_features = outputs.shape
        num_seq2seq_rows = num_patients * max_seq_length

        seq2seq_previous_treatments = np.zeros(
            (num_seq2seq_rows, max_seq_length, previous_treatments.shape[-1]))
        seq2seq_current_treatments = np.zeros(
            (num_seq2seq_rows, max_seq_length, current_treatments.shape[-1]))
        seq2seq_static_features = np.zeros(
            (num_seq2seq_rows, max_seq_length, static_features.shape[-1]))
        seq2seq_outputs = np.zeros(
            (num_seq2seq_rows, max_seq_length, outputs.shape[-1]))
        seq2seq_prev_outputs = np.zeros(
            (num_seq2seq_rows, max_seq_length, prev_outputs.shape[-1]))
        # seq2seq_vitals = np.zeros((num_seq2seq_rows, max_seq_length, vitals.shape[-1]))
        # seq2seq_next_vitals = np.zeros((num_seq2seq_rows, max_seq_length - 1, next_vitals.shape[-1]))
        seq2seq_active_entries = np.zeros(
            (num_seq2seq_rows, max_seq_length, active_entries.shape[-1]))
        seq2seq_sequence_lengths = np.zeros(num_seq2seq_rows)
        if 'stabilized_weights' in self.data:
            seq2seq_stabilized_weights = np.zeros(
                (num_seq2seq_rows, max_seq_length))

        total_seq2seq_rows = 0  # we use this to shorten any trajectories later

        for i in range(num_patients):
            sequence_length = int(sequence_lengths[i])

            for t in range(projection_horizon, sequence_length):  # shift outputs back by 1
                seq2seq_active_entries[total_seq2seq_rows, :(
                    t + 1), :] = active_entries[i, :(t + 1), :]
                if 'stabilized_weights' in self.data:
                    seq2seq_stabilized_weights[total_seq2seq_rows, :(
                        t + 1)] = stabilized_weights[i, :(t + 1)]
                seq2seq_previous_treatments[total_seq2seq_rows, :(
                    t + 1), :] = previous_treatments[i, :(t + 1), :]
                seq2seq_current_treatments[total_seq2seq_rows, :(
                    t + 1), :] = current_treatments[i, :(t + 1), :]
                seq2seq_outputs[total_seq2seq_rows, :(
                    t + 1), :] = outputs[i, :(t + 1), :]
                seq2seq_prev_outputs[total_seq2seq_rows, :(
                    t + 1), :] = prev_outputs[i, :(t + 1), :]
                seq2seq_static_features[total_seq2seq_rows, :(
                    t + 1), :] = static_features[i, :(t + 1), :]
                # seq2seq_vitals[total_seq2seq_rows, :(t + 1), :] = vitals[i, :(t + 1), :]
                # seq2seq_next_vitals[total_seq2seq_rows, :min(t + 1, sequence_length - 1), :] = \
                #     next_vitals[i, :min(t + 1, sequence_length - 1), :]
                seq2seq_sequence_lengths[total_seq2seq_rows] = t + 1
                # seq2seq_static_features[total_seq2seq_rows] = static_features[i]

                total_seq2seq_rows += 1

        # Filter everything shorter
        seq2seq_previous_treatments = seq2seq_previous_treatments[:total_seq2seq_rows, :, :]
        seq2seq_current_treatments = seq2seq_current_treatments[:total_seq2seq_rows, :, :]
        seq2seq_static_features = seq2seq_static_features[:total_seq2seq_rows, :]
        seq2seq_outputs = seq2seq_outputs[:total_seq2seq_rows, :, :]
        seq2seq_prev_outputs = seq2seq_prev_outputs[:total_seq2seq_rows, :, :]
        # seq2seq_vitals = seq2seq_vitals[:total_seq2seq_rows, :, :]
        # seq2seq_next_vitals = seq2seq_next_vitals[:total_seq2seqprocessed_rows, :, :]
        seq2seq_active_entries = seq2seq_active_entries[:total_seq2seq_rows, :, :]
        seq2seq_sequence_lengths = seq2seq_sequence_lengths[:total_seq2seq_rows]

        if 'stabilized_weights' in self.data:
            seq2seq_stabilized_weights = seq2seq_stabilized_weights[:total_seq2seq_rows]

        new_data = {
            'prev_treatments': seq2seq_previous_treatments,
            'current_treatments': seq2seq_current_treatments,
            'static_features': seq2seq_static_features,
            'prev_outputs': seq2seq_prev_outputs,
            'outputs': seq2seq_outputs,
            # 'vitals': seq2seq_vitals,
            # 'next_vitals': seq2seq_next_vitals,
            # 'unscaled_outputs': seq2seq_outputs * self.scaling_params['output_stds'] + self.scaling_params['output_means'],
            'sequence_lengths': seq2seq_sequence_lengths,
            'active_entries': seq2seq_active_entries,
        }
        if 'stabilized_weights' in self.data:
            new_data['stabilized_weights'] = seq2seq_stabilized_weights

        # self.data = new_data
        # self.exploded = True

        # data_shapes = {k: v.shape for k, v in self.data.items()}
        # logger.info(f'Shape of processed {self.subset_name} data: {data_shapes}')

        return new_data

    def get_exploded_dataset(self, dataset, min_length=1, only_active_entries=True, max_length=None):
        exploded_dataset = deepcopy(dataset)
        if max_length is None:
            max_length = max(exploded_dataset['sequence_lengths'][:])
        if not only_active_entries:
            exploded_dataset['active_entries'][:, :, :] = 1.0
            exploded_dataset['sequence_lengths'][:] = max_length
        # exploded_dataset.explode_trajectories(min_length)
        exploded_dataset = self.explode_trajectories(
            exploded_dataset, min_length)
        return exploded_dataset

    def forward(self, data):

        # data = self.get_exploded_dataset(data, 1, only_active_entries=True)
        # curr_treatments = data['current_treatments']
        # vitals_or_prev_outputs = []
        # # vitals_or_prev_outputs.append(data['vitals']) if self.has_vitals else None
        # # if self.autoregressive else None
        # vitals_or_prev_outputs.append(data['prev_outputs'])
        # vitals_or_prev_outputs = torch.cat(vitals_or_prev_outputs, dim=-1)
        # static_features = data['static_features']
        # outputs = data['outputs']

        # x = torch.cat((vitals_or_prev_outputs, curr_treatments), dim=-1)
        # x = torch.cat((x, static_features.unsqueeze(
        #     1).expand(-1, x.size(1), -1)), dim=-1)
        # x = torch.cat((x, outputs), dim=-1)
        # data = x
        # data = data.to(self.device)
        # print(f"Data shape: {data.shape}")
        # print(data)
        b, t, f = data.shape

        noise_mask = self.get_mask(data, self.strategy).to(self.device)
        # print(noise_mask[0])
        # print(data[0])
        noise = torch.randn((b, t, f)).to(self.device)
        noise = noise_mask * noise

        diffusion_t = torch.randint(0, self.diffusion_steps, (b, 1)).squeeze(1)
        alpha = self.alpha_torch[diffusion_t].unsqueeze(
            1).unsqueeze(2).to(self.device)

        noised_data = data * noise_mask
        noised_data = noised_data * (alpha**0.5) + noise * ((1 - alpha) ** 0.5)
        conditional_data = data * (1 - noise_mask)
        noised_data = noised_data + conditional_data
        noised_data = noised_data.float()

        predicted_noise = self.model_loop(
            noised_data.unsqueeze(3), noise_mask.unsqueeze(3), diffusion_t
        )
        predicted_noise = predicted_noise * noise_mask

        return (predicted_noise, noise, noise_mask)

    def eval_with_grad(self, data, scale=1):

        # with torch.no_grad():
        imputation_mask = self.get_mask(data, self.strategy).to(self.device)
        conditional_data = data * (1 - imputation_mask)
        random_noise = torch.randn_like(data) * imputation_mask * scale
        data_2 = (conditional_data + random_noise).unsqueeze(3)

        b, ti, f, e = data_2.shape
        imputed_samples = torch.zeros((b, ti, f)).to(self.device)
        x = conditional_data + random_noise

        for t in range(self.diffusion_steps - 1, -1, -1):

            x = x.unsqueeze(3).float()
            predicted_noise = self.model_loop(
                x, imputation_mask.unsqueeze(
                    3), torch.tensor([t]).to(self.device)
            )
            predicted_noise = predicted_noise * imputation_mask

            coeff1 = 1 / self.alpha_hat[t] ** 0.5
            coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5

            x = x.squeeze(3)
            x = coeff1 * (x - coeff2 * predicted_noise)

            if t > 0:
                noise = torch.randn_like(x)
                sigma = (
                    (1.0 - self.alpha[t - 1]) /
                    (1.0 - self.alpha[t]) * self.beta[t]
                ) ** 0.5
                x += sigma * noise

            x = data_2.squeeze(3) * (1 - imputation_mask) + x * imputation_mask

            imputed_samples = x

        return (imputed_samples, data, imputation_mask)

    def eval(
        self,
        data,
        imputation_mask,
        mean,
        std,
        scale=1,
        verbose=True,
        show_max_diff=False,
        show_rmse=False,
    ):

        conditional_data = data * (1 - imputation_mask)
        random_noise = torch.randn_like(data) * imputation_mask * scale
        data_2 = (conditional_data + random_noise).unsqueeze(3)

        b, ti, f, e = data_2.shape
        imputed_samples = torch.zeros((b, ti, f)).to(self.device)
        x = conditional_data + random_noise

        with torch.no_grad():

            for t in range(self.diffusion_steps - 1, -1, -1):

                x = x.unsqueeze(3).float()
                predicted_noise = self.model_loop(
                    x, imputation_mask.unsqueeze(
                        3), torch.tensor([t]).to(self.device)
                )
                predicted_noise = predicted_noise * imputation_mask

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5

                x = x.squeeze(3)
                x = coeff1 * (x - coeff2 * predicted_noise)

                if t > 0:
                    noise = torch.randn_like(x)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) /
                        (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    x += sigma * noise

                x = data_2.squeeze(3) * (1 - imputation_mask) + \
                    x * imputation_mask

            imputed_samples = x.detach()

        if show_max_diff == True:
            # show the data at torch.max(torch.abs(data[imputation_mask !=0] - imputed_samples[imputation_mask !=0]))
            print(
                "max difference = ",
                torch.max(
                    torch.abs(
                        data[imputation_mask != 0]
                        - imputed_samples[imputation_mask != 0]
                    )
                ).item(),
            )
            print(
                "data at max difference = ",
                data[imputation_mask != 0][
                    torch.argmax(
                        torch.abs(
                            data[imputation_mask != 0]
                            - imputed_samples[imputation_mask != 0]
                        )
                    )
                ].item(),
            )
            print(
                "imputed at max difference = ",
                imputed_samples[imputation_mask != 0][
                    torch.argmax(
                        torch.abs(
                            data[imputation_mask != 0]
                            - imputed_samples[imputation_mask != 0]
                        )
                    )
                ].item(),
            )

        mae = torch.mean(
            torch.abs(
                data[imputation_mask != 0] -
                imputed_samples[imputation_mask != 0]
            )
        ).item()
        if verbose == True:
            print("mae = ", mae)

        if show_rmse == True:
            # descale the data
            imputed_samples_copy = imputed_samples.detach().clone()
            imputed_samples_copy = imputed_samples_copy * std + mean
            data_copy = data.detach().clone()
            data_copy = data_copy * std + mean
            rmse = torch.sqrt(
                torch.mean(
                    (
                        data_copy[imputation_mask != 0]
                        - imputed_samples_copy[imputation_mask != 0]
                    )
                    ** 2
                )
            ).item()
            rmse = rmse / 1150 * 100
            print("rmse = ", rmse)
        # data_to_print = data[imputation_mask !=0]
        # imputed_samples_to_print = imputed_samples[imputation_mask !=0]
        # print("data:", data_to_print)
        # print("imputed:", imputed_samples_to_print)
        # print("absolute difference in the first 100 : ", torch.abs(data_to_print - imputed_samples_to_print)[:100])
        # print("mae = ", torch.mean(torch.abs(data_to_print - imputed_samples_to_print)).item())

        return (imputed_samples, data, imputation_mask, mae)

# # New main function for config handling and logging initialization
# def main(config: DictConfig):
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)
#     logger.info("Starting CausalDiff with config:\n%s", OmegaConf.to_yaml(config))

#     # Instantiate your model or application components using config args.
#     # Example:
#     # model = ModelLoop(
#     #     embed_dim=config.model.embed_dim,
#     #     diffusion_steps=config.model.diffusion_steps,
#     #     num_heads=config.model.num_heads,
#     #     kernel_size=config.model.kernel_size,
#     #     ff_dim=config.model.ff_dim,
#     #     num_cells=config.model.num_cells,
#     #     dropout=config.model.dropout,
#     #     num_residual_layers=config.model.num_residual_layers,
#     #     method=config.model.method,
#     #     device=config.device,
#     # )
#     # ...additional logic for training/evaluation...

# # Main guard to load config and start main()
# if __name__ == "__main__":
#     import sys
#     # Load config from the provided YAML file or use a default path
#     config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/causaldiff.yaml"
#     config = OmegaConf.load(config_path)
#     main(config)
