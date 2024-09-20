import torch
from tfcl.utils.get_norm import * 
from tfcl.nn.layers.conv2Dnormed import * 
from tfcl.nn.layers.ptattention import * 
from tfcl.nn.layers.patchattention_thw import  *

class FusionV2(torch.nn.Module):
    def __init__(self,nfilters_in, nfilters_out, spatial_size, scales, kernel_size=3, padding=1,nheads=1, norm = 'BatchNorm', norm_groups=None,correlation_method='linear'):
        super().__init__()
        
        self.relatt = RelPatchAttention2D(in_channels=nfilters_in, out_channels=nfilters_out, kernel_size=kernel_size, padding=padding, nheads=nheads, norm =norm, norm_groups=norm_groups, spatial_size=spatial_size, scales=scales,  correlation_method=correlation_method)
        self.fuse = Conv2DNormed(in_channels=nfilters_out*2, out_channels=nfilters_out,kernel_size= kernel_size, padding = padding, norm_type= norm, num_groups=norm_groups, groups=nheads)

    def forward(self, input_t1, input_t2):

        # Enhanced output of 1, based on memory of 2
        out12 = input_t1 + self.relatt(input_t2,input_t1)
        # Enhanced output of 2, based on memory of 1
        out21 = input_t2 + self.relatt(input_t1,input_t2)

        fuse = self.fuse(torch.cat([out12, out21],dim=1))
        fuse = torch.relu(fuse)

        return fuse

class FusionV23D(torch.nn.Module):
    """
    In this version, I ditch gamma params, for starting at zero and i balance that by very small initialization on D2Sigmoid
    Also I do not anymore multiply attention output with input, because attention is in [-0.1,0.1] if not scaled, [-1,1]
    """
    def __init__(self,nfilters_in, nfilters_out, scales, kernel_size=3, padding=1,nheads=1, norm = 'BatchNorm', norm_groups=None,correlation_method='mean' ,  TimeDim=None, depth=10.0):
        super().__init__()
        
        self.relatt = RelPatchAttention3DTCHW(in_channels=nfilters_in, out_channels=nfilters_out, kernel_size=kernel_size, padding=padding, nheads=nheads, norm =norm, norm_groups=norm_groups, scales=scales,  correlation_method=correlation_method,TimeDim=TimeDim, depth=depth)
        self.fuse = Conv3DNormed(in_channels=nfilters_out*2, out_channels=nfilters_out,kernel_size= kernel_size, padding = padding, norm_type= norm, num_groups=norm_groups, groups=nheads)

    def forward(self, input_t1, input_t2):

        # Enhanced output of 1, based on memory of 2
        out12 = input_t1 + self.relatt(input_t2,input_t1)
        # Enhanced output of 2, based on memory of 1
        out21 = input_t2 + self.relatt(input_t1,input_t2)

        fuse = self.fuse(torch.cat([out12, out21],dim=1))
        fuse = torch.relu(fuse)

        return fuse


