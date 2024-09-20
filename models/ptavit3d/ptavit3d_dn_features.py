import torch
import numpy as np

from tfcl.nn.layers.conv3Dnormed import *


from tfcl.nn.layers.patchattention_thw import  *  
from tfcl.nn.layers.scale import *
from tfcl.nn.layers.combine import *
from tfcl.nn.units.ptavit3d import *


import copy

class FusionCAT(torch.nn.Module):
    def __init__(self,nfilters_in,nfilters_out, nheads, kernel_size=3, padding=1, norm = 'BatchNorm', norm_groups=None):
        super().__init__()
        
        self.fuse = Conv3DNormed(in_channels=nfilters_in*2, out_channels=nfilters_out,kernel_size= kernel_size, padding = padding, norm_type= norm, num_groups=norm_groups, groups=nheads)

    def forward(self, out12, out21):

        fuse = self.fuse(torch.cat([out12, out21],dim=1))
        fuse = torch.relu(fuse)

        return fuse


class FuseHiLo(torch.nn.Module):
    def __init__(self, nfilters, nfilters_embed=96, scales=(4,8,8),   norm_type = 'BatchNorm', norm_groups=None,depth=10.0):
        super().__init__()



        self.embedding1 = Conv3DNormed(in_channels = nfilters, out_channels = nfilters_embed, kernel_size = 1, padding=0, norm_type=norm_type, num_groups=norm_groups)
        self.embedding2 = Conv3DNormed(in_channels = nfilters, out_channels = nfilters_embed, kernel_size = 1, padding=0, norm_type=norm_type, num_groups=norm_groups)


        self.upscale = UpSample2D3D(in_channels=nfilters_embed,out_channels=nfilters_embed,scale_factor=4,norm_type=norm_type,norm_groups=norm_groups)


        self.conv3d = Conv3DNormed(in_channels=nfilters_embed*2, out_channels = nfilters_embed,kernel_size =1, norm_type=norm_type, num_groups=norm_groups)
        self.att = RelPatchAttention3DTCHW(in_channels=nfilters_embed, out_channels = nfilters_embed,nheads=nfilters_embed//4,norm=norm_type,norm_groups=norm_groups,
                scales=scales,
                depth=depth
              )


    def forward(self, UpConv4, conv1):
        # conv1: full resolution
        # UpConv4: 1/4 of original resolution 

        UpConv4 = self.embedding1(UpConv4)
        UpConv4 = self.upscale(UpConv4)
        conv1   = self.embedding2(conv1)

        # second last layer 
        convl = torch.cat([conv1,UpConv4],dim=1)
        #print(convl.shape)
        conv = self.conv3d(convl)
        conv = torch.relu(conv)

        # Apply attention
        conv = conv * (1.+self.att(conv,conv))

        return conv




class ptavit3d_dn_features(torch.nn.Module):
    def __init__(self,  in_channels, spatial_size_init, nfilters_init=96, nfilters_embed=32, nheads_start=96//4, depths=[2,2,5,2], verbose=True, norm_type='GroupNorm', norm_groups=4, correlation_method='mean', TimeDim=None, attention_depth=0.0,stem_norm=True):
        super().__init__()
 

        # NEW, EXPRIMENT WITH
        def closest_power_of_2(num_array):
            log2_array = np.log2(num_array)
            rounded_log2_array = np.round(log2_array)
            closest_power_of_2_array = np.power(2, rounded_log2_array)
            return np.maximum(closest_power_of_2_array, 1).astype(int)


        def resize_scales(channel_size, spatial_size, scales_all):
            temp = np.array(scales_all)*np.array([channel_size/96,spatial_size[0]/256,spatial_size[1]/256])
            return closest_power_of_2(temp).tolist()


        scales_all = [[16,16,16],[32,8,8],[64,4,4],[128,2,2],[128,2,2],[128,1,1],[256,1,1],[256,1,1]] # DEFAULT, nice results 
        scales_all = resize_scales(nfilters_init, spatial_size_init,scales_all)
        self.scales_all = scales_all


        self.depth = depth = len(depths)
        num_stages = len(depths)
        dims = tuple(map(lambda i: (2 ** i) * nfilters_init, range(num_stages)))
        dims = (nfilters_init, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        # TODO Make it optional causal 
        self.conv1     = Conv3DNormed(in_channels=in_channels, out_channels = nfilters_init, kernel_size=1,padding=0,strides=1, norm_type=norm_type, num_groups=norm_groups)


        # Scale 1/2 
        if stem_norm:
            self.conv_stem = nn.Sequential(
            nn.Conv3d( nfilters_init, nfilters_init, (3,3,3), stride = (1,2,2), padding = (1,1,1)),
            nn.GroupNorm(num_groups=norm_groups, num_channels=nfilters_init),
            nn.Conv3d(nfilters_init, nfilters_init, 3, padding = 1),
            nn.GroupNorm(num_groups=norm_groups, num_channels=nfilters_init)
            )
        else:
            self.conv_stem = nn.Sequential(
            nn.Conv3d( nfilters_init, nfilters_init, (3,3,3), stride = (1,2,2), padding = (1,1,1)),
            nn.Conv3d(nfilters_init, nfilters_init, 3, padding = 1)
            )

        # The stem scales 1/2, the PTAViTFD layers scale 1/2 in first MBConv
        spatial_size_init = tuple(ts // 4 for ts in spatial_size_init)

        # List of convolutions and pooling operators 
        self.stages_dn = [] 
        self.fuse     = [] 
        self.atts_fuse     = [] 

        if verbose:
            print (" @@@@@@@@@@@@@ Going DOWN @@@@@@@@@@@@@@@@@@@ ")
        for idx, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depths)):
        #for idx,  layer_depth in enumerate(depths):
            nheads = nheads_start * 2**idx #
            scales = scales_all[idx]
            spatial_size = tuple( ts // 2**idx for ts in spatial_size_init )


            #print ("drop_path istart::{}, iend::{}".format(dp_istart,dp_iend))
            if verbose:
                print ("depth:= {0}, layer_dim_in: {1}, layer_dim: {2}, stage_depth::{3}, spatial_size::{4}, scales::{5}".format(idx,layer_dim_in,layer_dim,layer_depth,spatial_size, scales)) 

            self.stages_dn.append(PTAViTStage3DTCHW(
                layer_dim_in=layer_dim_in,
                layer_dim=layer_dim,
                layer_depth=layer_depth,
                nheads=nheads,
                scales=scales,
                downsample=True,
                mbconv_expansion_rate = 4,
                mbconv_shrinkage_rate = 0.25,
                dropout = 0.1,
                correlation_method=correlation_method,
                TimeDim=TimeDim,
                depth=attention_depth
                ))

            self.fuse.append( FusionCAT( 
                nfilters_in         =   layer_dim, 
                nfilters_out        =   layer_dim,  
                nheads              =   nheads,
                norm                =   norm_type, 
                norm_groups         =   norm_groups)  
                )

            self.atts_fuse.append(RelPatchAttention3DTCHW(
                in_channels         =   layer_dim, 
                out_channels        =   layer_dim, 
                kernel_size         =   3, 
                padding             =   1, 
                nheads              =   nheads, 
                norm                =   norm_type, 
                norm_groups         =   norm_groups, 
                scales              =   scales, 
                correlation_method  =   correlation_method,
                TimeDim		    =	TimeDim,
                depth		    =   attention_depth))


        self.stages_dn = torch.nn.ModuleList(self.stages_dn)
        self.fuse     = torch.nn.ModuleList(self.fuse)
        self.atts_fuse     = torch.nn.ModuleList(self.atts_fuse)


        self.stages_up = [] 
        self.UpCombs  = [] 

        # Reverse order, ditch first 
        dim_pairs = dim_pairs[::-1]
        depths    = depths[::-1]
        
        dim_pairs = dim_pairs[:-1]
        #depths    = depths[:-1]# works 
        depths    = depths[1:]

        

        if verbose:
            print (" XXXXXXXXXXXXXXXXXXXXX Coming up XXXXXXXXXXXXXXXXXXXXXXXXX " )

        for idx, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depths)):
        #for idx, layer_depth in enumerate(depths):
            idx = len(depths)-1 - idx

            #layer_dim_in = layer_dim = nfilters_init * 2 **(idx)
            nheads = int(nheads_start * 2**(idx)) #

            # XXXXXXXXXXXX Possible bug here XXXXXXXXXXXXXXX
            spatial_size = tuple( ts // 2**idx for ts in spatial_size_init )
            # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            scales = scales_all[idx]

            if verbose:
                print ("depth:= {0}, layer_dim_in: {1}, layer_dim: {2}, stage_depth::{3}, spatial_size::{4}, scales::{5}".format(2*depth-idx-2, 
                    layer_dim_in, layer_dim_in, layer_depth,spatial_size, scales))



            self.stages_up.append(PTAViTStage3DTCHW(
                layer_dim_in	      = layer_dim_in,
                layer_dim	      = layer_dim_in,
                layer_depth	      = layer_depth,
                nheads		      = nheads,
                scales		      = scales,
                downsample	      = False,
                mbconv_expansion_rate = 4,
                mbconv_shrinkage_rate = 0.25,
                dropout 	      =	0.1,
                correlation_method    =	correlation_method,
                TimeDim		      =	TimeDim,
                depth                 = attention_depth
                ))


            self.UpCombs.append(combine_layers3D(
                layer_dim_in, 
                norm_type=norm_type,
                norm_groups=norm_groups))

        
        self.stages_up   = torch.nn.ModuleList(self.stages_up)
        self.UpCombs    = torch.nn.ModuleList(self.UpCombs)

        # The first option worked well, but now I am thinking the scales may not be optimal, and i need better fusion policy 
        self.fuse_hi_lo = FuseHiLo( nfilters=layer_dim_in, nfilters_embed=nfilters_embed, scales=(4,8,8),   norm_type = norm_type, norm_groups=norm_groups,depth=attention_depth)
        #self.fuse_hi_lo = FuseHiLo( nfilters=layer_dim_in, nfilters_embed=nfilters_embed, spatial_size=spatial_size,scales=scales_all[0],   norm_type = norm_type, norm_groups=norm_groups)



    def forward(self, input_t1):
        conv1_t1 = self.conv1(input_t1)
            
	# Reduce spatial size 
        conv1 = self.conv_stem(conv1_t1)
        
        # ******** Going down ***************
        #print ("Going DOWN")
        fusions   = []
        for idx in range(self.depth):
            #print ("Down index::{}".format(idx))
            #print ("Shapes BEFORE stages::{}, {}".format(conv1.shape, conv2.shape))
            conv1 = self.stages_dn[idx](conv1)

            # Evaluate fusions 
            fusions = fusions + [conv1]
            #print ("Fusion shape::{}".format(fusions[-1].shape))
            #print (" ********************************************************** ")

        #print ("XXXXXXXXXXXXXXXXXXXXXXXXXX")
        #print ("Coming UP")
        # ******* Coming up ****************
        convs_up = fusions[-1]
        # @@@@@@@@@@@@@@@@@@@@@@@@@@ POTENTIAL ERROR @@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        convs_up = torch.relu(convs_up) # middle of network activation 
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        
        for idx in range(self.depth-1):
        #for idx in range(self.depth):
            #print ("Shapes BEFORE stages::{}, {}".format(convs_up.shape, fusions[-idx-2].shape))
            convs_up = self.UpCombs[idx](convs_up, fusions[-idx-2])
            #print ("Shapes AFTER COMBINE::{}".format(convs_up.shape))
            convs_up = self.stages_up[idx](convs_up)
            #print ("Shapes AFTER stage::{}".format(convs_up.shape))
            #print (" ==============================================  ")
         


        final = self.fuse_hi_lo(convs_up, conv1_t1)
        return final 
    

