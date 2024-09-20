# Modified with our attention from https://github.com/lucidrains/vit-pytorch

from functools import partial
import numpy as np

import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

# helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# helper classes
class LayerNormChannelsLast(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, input):
        x = rearrange(input,'b c h w-> b h w c')
        x = self.norm(x)
        x = rearrange(x,'b h w c -> b c h w')
        return x

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNormChannelsLast(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x

# @@@@@@@@@@@@@@@@@ SSG2 CUSTOM @@@@@@@@@@@@@@@@@@@@@@@@
class PreNormResidualAtt(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNormChannelsLast(dim)
        self.fn = fn

    def forward(self, x):
        return (self.fn(self.norm(x)) + 1)*x
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x = rearrange(x,'b c h w -> b h w c')
        x = self.net(x)
        return rearrange(x,'b h w c -> b c h w')

# MBConv
class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x

class Dropsample(nn.Module):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob
  
    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)

def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride = stride, padding = 1, groups = hidden_dim),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate = shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        nn.BatchNorm2d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout = dropout)

    return net


# @@@@@@@@@@@@@@@@@ SSG2 CUSTOM @@@@@@@@@@@@@@@@@@@@@@@@
# D
from tfcl.nn.layers.ptattention import * 
class PTAttention(nn.Module):
    def __init__(
        self,
        dim,
        nheads = 32,
        dropout = 0.,
        scales = (4,4),
        spatial_size=64,
        verbose=False,
        correlation_method='linear',
    ):
        super().__init__()

        spatial_size = spatial_size
        if verbose:
            print("nfilters::{}, spatial_size::{}, scales::{}, nheads::{}".format(dim, spatial_size,scales,nheads))

        self.att = RelPatchAttention2D(dim,dim,spatial_size,scales=scales,norm='GroupNorm',norm_groups=dim//4,nheads=nheads,correlation_method=correlation_method)

    def forward(self,input):       
        return   self.att(input,input) # [-0.1,0.1] with d2sigmoid


class PTAViTStage(nn.Module):
    def __init__(
        self,
        layer_dim_in,
        layer_dim,
        layer_depth,
        nheads,
        scales,
        spatial_size,
        downsample=False,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1,
        correlation_method='linear',
    ):
        super().__init__()

        if downsample:
            spatial_size = spatial_size//2

        stage = []
        for stage_ind in range(layer_depth):
            is_first = stage_ind == 0
            # This gives nans, don't know why at the moment, getting there ...               
            block = nn.Sequential(
                MBConv(
                    layer_dim_in if is_first else layer_dim,
                    layer_dim,
                    downsample = downsample if is_first else False,
                    expansion_rate = mbconv_expansion_rate,
                    shrinkage_rate = mbconv_shrinkage_rate
                ),
                PreNormResidualAtt(layer_dim, PTAttention(dim = layer_dim, nheads = nheads, dropout = dropout,spatial_size=spatial_size,
                    scales=scales,
                    correlation_method=correlation_method,
                    )),
                PreNormResidual(layer_dim, FeedForward(dim = layer_dim, dropout = dropout))
            )


            stage.append(block)
        self.stage = torch.nn.Sequential(*stage)
    def forward(self,input):
        return self.stage(input)



class PTAViTStage_no_down(nn.Module):
    def __init__(
        self,
        layer_dim_in,
        layer_dim,
        layer_depth,
        nheads,
        scales,
        spatial_size,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1,
        correlation_method='linear',
        norm_type='BatchNorm',
        norm_groups=None
    ):
        super().__init__()


        stage = []
        for stage_ind in range(layer_depth):
            # This gives nans, don't know why at the moment, getting there ...               
            block = nn.Sequential(
                MBConv(
                    layer_dim_in, 
                    layer_dim_in,
                    downsample = False,
                    expansion_rate = mbconv_expansion_rate,
                    shrinkage_rate = mbconv_shrinkage_rate
                ),
                PreNormResidualAtt(layer_dim_in, PTAttention(dim = layer_dim_in, nheads = nheads, dropout = dropout,spatial_size=spatial_size,
                    scales=scales,
                    correlation_method=correlation_method,
                    )),
                PreNormResidual(layer_dim_in, FeedForward(dim = layer_dim_in, dropout = dropout))
            )


            stage.append(block)
        self.stage = torch.nn.Sequential(*stage)
    def forward(self,input):
        return self.stage(input)
