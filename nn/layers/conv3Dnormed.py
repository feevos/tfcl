import torch
from tfcl.utils.get_norm import * 


class Conv3DNormed(torch.nn.Module):
    """
        Convenience wrapper layer for 3D convolution followed by a normalization layer 
        All other keywords are the same as gluon.nn.Conv3D 
    """

    def __init__(self, in_channels, out_channels, kernel_size, strides=(1, 1,1),
                 padding=(0, 0, 0), dilation=(1, 1, 1), norm_type = 'BatchNorm', num_groups=None, 
                 groups=1):
        super(Conv3DNormed,self).__init__()

        self.conv3d = torch.nn.Conv3d(in_channels=in_channels, 
                                      out_channels=out_channels,
                                      kernel_size= kernel_size,
                                      stride= strides,
                                      padding=padding,
                                      dilation= dilation,
                                      bias=False,
                                      groups=groups)
        self.norm_layer = get_norm3d(name=norm_type,channels=out_channels,num_groups=num_groups)

    @torch.jit.export
    def forward(self,input:torch.Tensor):

        x = self.conv3d(input)
        x = self.norm_layer(x)

        return x

