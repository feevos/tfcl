import torch

from tfcl.nn.layers.scale import *
from tfcl.nn.layers.conv2Dnormed import *
from tfcl.nn.layers.conv3Dnormed import *




class combine_layers(torch.nn.Module):
    def __init__(self,nfilters,  norm_type = 'BatchNorm', norm_groups=None):
        super(combine_layers,self).__init__()


        # This performs convolution, no BatchNormalization. No need for bias.
        self.up = UpSample2D(in_channels=nfilters*2, out_channels=nfilters, norm_type = norm_type, norm_groups=norm_groups)

        self.conv_normed = Conv2DNormed(in_channels = 2*nfilters,out_channels=nfilters,
                                            kernel_size=(1,1),
                                            padding=(0,0),
                                            norm_type=norm_type,
                                            num_groups=norm_groups)




    def forward(self,_layer_lo, _layer_hi):
        up = self.up(_layer_lo)

        up = torch.relu(up)
        x = torch.cat([up,_layer_hi], dim=1)
        x = self.conv_normed(x)

        return x

class combine_layers3D(torch.nn.Module):
    def __init__(self,nfilters,  norm_type = 'BatchNorm', norm_groups=None,causal=False):
        super().__init__()


        # This performs convolution, no BatchNormalization. No need for bias.
        self.up = UpSample2D3D(in_channels=nfilters*2, out_channels=nfilters, norm_type = norm_type, norm_groups=norm_groups, causal=causal)

        self.conv_normed = Conv3DNormed(in_channels = 2*nfilters,out_channels=nfilters,
                                            kernel_size=(1,1,1),
                                            padding=(0,0,0),
                                            norm_type=norm_type,
                                            num_groups=norm_groups)




    def forward(self,_layer_lo, _layer_hi):
        up = self.up(_layer_lo)

        up = torch.relu(up)
        x = torch.cat([up,_layer_hi], dim=1)
        x = self.conv_normed(x)

        return x


