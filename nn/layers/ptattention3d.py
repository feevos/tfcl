import torch
from torch import einsum
import numpy as np 

class PatchifyTHW(torch.nn.Module):
    def __init__(self, hscale, wscale):
        super().__init__()
        self.h = hscale
        self.w = wscale
        self.unfold_shape = None

    def _2patch(self,input):
        shape = input.shape # B x T x C x H x W
        h     = torch.div(shape[-2], self.h, rounding_mode='floor')
        w     = torch.div(shape[-1], self.w, rounding_mode='floor')


        # Currently only works with stride = window size
        sh    = h 
        sw    = w 

        # Here I assume stride is equal to c
        patch = input.unfold(-2,h,sh).unfold(-2,w,sw).permute(0,1,3,4,2,-2,-1).contiguous()
        self.unfold_shape = patch.shape
        return patch

    def _2tensor(self, patch):
        output_h    = self.unfold_shape[2] * self.unfold_shape[5]
        output_w    = self.unfold_shape[3] * self.unfold_shape[6]

        tensorpatch = patch.permute(0, 1, 4, 2, 5, 3, 6).contiguous()

        tensorpatch = tensorpatch.view(self.unfold_shape[0],self.unfold_shape[1],self.unfold_shape[4],output_h,output_w)
        return tensorpatch


class BASE_RelPatchAttention2D_THW(torch.nn.Module):
    def __init__(self, nfilters, scales=(16,16)):
        super().__init__()                                                                                                                                                            


        self.scales = scales
        self.patchify = PatchifyTHW(hscale=scales[0],wscale=scales[1])                                                                                               

    def qk_sim(self,q,k,smooth=1.e-5):
        # q --> B, (t, h, w), C, H/h, W/w 
        # k --> B, [t, h, w], C, H/h, W/w                                                                                                                                                                                                                                                                                          

        qk = einsum('bjklmno,bstrmno->bjklstr',q,k) #B, (t, h, w), [t, h, w]
        qq = einsum('bjklmno,bjklmno->bjkl',q,q)    #B, (t, h, w)  
        kk = einsum('bstrmno,bstrmno->bstr',k,k)    #B, [t, h, w]


        denum = (qq[:,:,:,:,None,None,None]+kk[:,None,None,None])-qk +smooth
        result = (qk+smooth) * denum.reciprocal()


        return result

    # Mean operation instead of linear allows for arbitrary length time series as input 
    def qk_compact(self,qk):
        # input qk : B x (t*h*w) x [t x h x w] 
        # output 4 v : B x [t x h x w] 
        tqk = qk.mean(dim=1) #permute(0,2,3,4,1)

        return tqk

    def qk_select_v(self,qk,vpatch, smooth=1.e-5):

        # qk --> B, t, h, w, t, h, w 
        # v  --> B , t, h, w, C, H/h, W/w 
        # qk: similarity between q and k values 
        # v : values 
        # qkv: v values emphasized where qk says it's important 

        # Options 1, 2 and 3 unified in this method 
        tqk = qk.reshape(qk.shape[0],-1,*qk.shape[4:]) # B x (t*h*w) x [ t x h x w]
        tqk =  self.qk_compact(tqk) # B x [c x h x w]
        qkvv = einsum('brst, brstmno -> brstmno', tqk,vpatch)


        qkvv = self.patchify._2tensor(qkvv)
        return qkvv


    def get_att(self, q,k,v):
        # q,k,v --> B x T x C x H x W  
                
        # =======================================================
        qp = self.patchify._2patch(q) # B, T, h, w, C, H//h, W//w  
        kp = self.patchify._2patch(k) # B, T, h, w, C, H//h, W//w  
        vp = self.patchify._2patch(v) # B, T, h, w, C, H//h, W//w  
        # =======================================================
        


        qpkp = self.qk_sim(qp,kp) # 
        vout    = self.qk_select_v(qpkp,vp)  # 

        return vout



from tfcl.nn.activations.d2sigmoid import *
from tfcl.nn.layers.conv3Dnormed import *
class RelPatchAttentionTHW(torch.nn.Module):
    # Fastest implementation so far  - with sigmoid
    def __init__(self,in_channels,out_channels, scales=(16,16), kernel_size=3,padding=1,nheads=1,norm='BatchNorm',norm_groups=None):
        super().__init__()


        self.act =   D2Sigmoid()

        self.patch_attention = BASE_RelPatchAttention2D_THW(out_channels,  scales)

        self.query   = Conv3DNormed(in_channels=in_channels,out_channels=out_channels,kernel_size= kernel_size, padding = padding, norm_type= norm, num_groups=norm_groups, groups=nheads)
        self.kv      = Conv3DNormed(in_channels=in_channels,out_channels=out_channels*2,kernel_size= kernel_size, padding = padding, norm_type= norm, num_groups=norm_groups, groups=nheads*2)


    def forward(self,input1:torch.Tensor, input2:torch.Tensor):
        # Best configuration so far to avoid nans
        q    = self.query(input1) # B,C,T,H,W
        k,v  = self.kv(input2).split(q.shape[1],1) # B,C,T,H,W

        q    = self.act(q)
        k    = self.act(k)

        q = q.permute(0,2,1,3,4)
        k = k.permute(0,2,1,3,4)
        v = v.permute(0,2,1,3,4)


        v    = self.patch_attention.get_att(q,k,v)
        v    = self.act(v) 


        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


        return v

