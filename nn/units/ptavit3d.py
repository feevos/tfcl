# Modified with my attention from https://github.com/lucidrains/vit-pytorch

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

class LayerNormChannelsLast3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, input):
        x = rearrange(input,'b c s h w-> b s h w c')
        x = self.norm(x)
        x = rearrange(x,'b s h w c -> b c s h w')
        return x

class PreNormResidual3D(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNormChannelsLast3D(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class PreNormResidualAtt3D(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNormChannelsLast3D(dim)
        self.fn = fn

    def forward(self, x):
        return (self.fn(self.norm(x)) + 1)*x

class FeedForward3D(nn.Module):
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
        x = rearrange(x,'b c s h w -> b s h w c')
        x = self.net(x)
        return rearrange(x,'b s h w c -> b c s h w')

# MBConv
# This SE module is like self attention kind, the self.gate(x) is the "attention" on x. 
# including rescaling
class SqueezeExcitation3D(nn.Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)
        # https://arxiv.org/pdf/1709.01507.pdf 

        self.gate = nn.Sequential(
            Reduce('b c s h w -> b s c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(), # CANDIDATE FOR D2S
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(), # CANDIDATE FOR D2S
            Rearrange('b s c -> b c s 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual3D(nn.Module):
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

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)



def MBConv3D(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.,
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = (1,2,2) if downsample else (1,1,1)

    conv3d_spatial = nn.Conv3d(hidden_dim, hidden_dim, (3,3,3), stride = stride, padding = (1,1,1), groups = hidden_dim)


    net = nn.Sequential(
        nn.Conv3d(dim_in, hidden_dim, 1),
        nn.BatchNorm3d(hidden_dim),
        nn.GELU(),
        conv3d_spatial,
        nn.BatchNorm3d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation3D(hidden_dim, shrinkage_rate = shrinkage_rate),
        nn.Conv3d(hidden_dim, dim_out, 1),
        nn.BatchNorm3d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual3D(net, dropout = dropout)

    return net






from tfcl.nn.layers.patchattention_thw import  *
class PTAttention3DTCHW(nn.Module):
    def __init__(
        self,
        dim,
        nheads = 32,
        dropout = 0.,
        scales = (4,4),
        verbose=False,
        correlation_method='mean',
        TimeDim=None,
        depth=10.0
    ):
        super().__init__()

        if verbose:
            print("nfilters::{}, scales::{}, nheads::{}".format(dim, scales,nheads))
        self.att      =  RelPatchAttention3DTCHW(
                                in_channels  	   = dim,
                                out_channels 	   = dim,
                                nheads       	   = nheads,
                                scales       	   = scales,
                                norm         	   = 'GroupNorm',
                                norm_groups  	   = dim//4,
                                correlation_method = correlation_method,
                                TimeDim            = TimeDim, 
                                depth 		   = depth)

    def forward(self,input):       
        #return  input * self.att(input,input) # This hasn't been proven to be super good 
        return   self.att(input,input) # [-1,1] with d2sigmoid




class PTAViTStage3DTCHW(nn.Module):
    def __init__(
        self,
        layer_dim_in,
        layer_dim,
        layer_depth,
        nheads,
        scales,
        downsample=False,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1,
        correlation_method='mean',
        TimeDim=None,
        depth=10.0
    ):
        super().__init__()


        stage = []
        for stage_ind in range(layer_depth):
            is_first = stage_ind == 0
            block = nn.Sequential(
                MBConv3D(
                    layer_dim_in if is_first else layer_dim,
                    layer_dim,
                    downsample = downsample if is_first else False,
                    expansion_rate = mbconv_expansion_rate,
                    shrinkage_rate = mbconv_shrinkage_rate,
                ),
                # XXXXXX TimeSpatial Attention XXXXX
                PreNormResidualAtt3D(
                    layer_dim, 
                    PTAttention3DTCHW(
                        dim = layer_dim, 
                        nheads = nheads, 
                        dropout = dropout,
                        scales=scales,
                        correlation_method=correlation_method,
                        TimeDim=TimeDim,
                        depth=depth)),
                PreNormResidual3D(
                    layer_dim, 
                    FeedForward3D(
                        dim = layer_dim, 
                        dropout = dropout))
            )
            stage.append(block)

        self.stage = torch.nn.Sequential(*stage)
    def forward(self,input):
        return self.stage(input)





