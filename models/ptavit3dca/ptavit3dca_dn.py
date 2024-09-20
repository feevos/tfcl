import torch
import sys
sys.path.append(r'../../../')
from tfcl.models.ptavit3dca.ptavit3dca_dn_features import *
from tfcl.models.head_cmtsk import head_cmtsk 

class Lambda(nn.Module):
    def __init__(self,  fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) 

# Mantis conditioned multitasking. 
class ptavit3dca_dn(torch.nn.Module):
    def __init__(self, in_channels_s2, in_channels_s1, NClasses, nfilters_init=96, nfilters_embed=96, nheads_start=96//4, depths=[2,2,5,2], spatial_size_init=(128,128), verbose=True, norm_type='GroupNorm', norm_groups=4,  correlation_method='mean',segm_act='sigmoid',TimeDim=None, nblocks3d=1,attention_depth=10.0):
        super().__init__()
                   
        self.features = ptavit3d_ca_dn_features(in_channels_s2 = in_channels_s2,  in_channels_s1 = in_channels_s1, spatial_size_init=spatial_size_init, nfilters_init=nfilters_init, nfilters_embed=nfilters_embed, nheads_start = nheads_start, depths = depths, verbose=verbose, norm_type=norm_type, norm_groups=norm_groups, correlation_method=correlation_method,TimeDim=TimeDim,attention_depth=attention_depth)

        scales = self.features.scales_all[0]
        nblocks3d=nblocks3d
        self.head3D = torch.nn.Sequential(
            PTAViTStage3DTCHW(
                layer_dim_in=nfilters_embed,                
                layer_dim=nfilters_embed,                   
                layer_depth=nblocks3d,                 
                nheads=nfilters_embed//4,         
                scales=scales,                      
                downsample=False,            
                mbconv_expansion_rate = 4,   
                mbconv_shrinkage_rate = 0.25,
                dropout = 0.1,               
                correlation_method='mean',   
                TimeDim=TimeDim,                
                depth=attention_depth),
            	Lambda(lambda x: x.mean(dim=2)),
            	head_cmtsk(nfilters=nfilters_init, nfilters_embed=nfilters_embed,NClasses=NClasses,
                	norm_type=norm_type,norm_groups=norm_groups)
        )


    # Standard Transformer stuff, predicts only 1 ahead 
    def forward(self,input_t1,input_t2):

        features3D = self.features(input_t1,input_t2)
        preds2D3D  = self.head3D(features3D)

        return preds2D3D 

