import torch
from torch import einsum
import numpy as np 


from tfcl.nn.activations.d2sigmoid import *
from tfcl.nn.layers.conv3Dnormed import *
from tfcl.nn.layers.conv2Dnormed import *


# This is non-overlapping patch attention, 
# Pros: Can be applied in chw space, 
# Cons: non-overlapping!!! 
class Patchify3DCHW(torch.nn.Module):
    def __init__(self, cscale, hscale, wscale):
        super().__init__()
        self.c = cscale
        self.h = hscale
        self.w = wscale
        self.unfold_shape = None

    def _2patch(self,input):
        shape = input.shape # B x T x C x H x W
        c     = torch.div(shape[-3], self.c, rounding_mode='floor')
        h     = torch.div(shape[-2], self.h, rounding_mode='floor')
        w     = torch.div(shape[-1], self.w, rounding_mode='floor')


        # Currently only works with stride = window size
        sc    = c
        sh    = h # // 2
        sw    = w # // 2

        # Here I assume stride is equal to c
        #patch = input.unfold(-2,h,sh).unfold(-2,w,sw).permute(0,1,3,4,2,-2,-1).contiguous()
        patch = input.unfold(2,c,sc).unfold(3,h,sh).unfold(4,w,sw)
        self.unfold_shape = patch.shape
        return patch

    def _2tensor(self, patch):
        output_c    = self.unfold_shape[2] * self.unfold_shape[5]
        output_h    = self.unfold_shape[3] * self.unfold_shape[6]
        output_w    = self.unfold_shape[4] * self.unfold_shape[7]

        tensorpatch = patch.permute(0, 1, 2, 5, 3, 6,4,7).contiguous()

        tensorpatch = tensorpatch.view(self.unfold_shape[0],self.unfold_shape[1],output_c, output_h,output_w)
        return tensorpatch


# This version has different summation indices, in particular it sums the Time index of the q chw 
class BASE_RelPatchAttention3D_TCHW(torch.nn.Module):
    # This one does comparison of all times with all times using mask 
    def __init__(self, nfilters, scales, correlation_method='sum',TimeDim=None, depth=0.0):
                 #, position_encoding=True, height=None, width=None):                                                                                                                      
        super().__init__()                                                                                                   

        self.alpha = 2.0**depth
        self.beta  = 2.0*self.alpha-1                                                                                                                                                                             
        if depth==0.0:
            self.qk_sim= self._qk_identity_sim_v1
        else:
            self.qk_sim= self._qk_identity_sim_v2



        self.scales = scales
        self.patchify = Patchify3DCHW(cscale=scales[0],hscale=scales[1],wscale=scales[2]) 
        
       
    
        if correlation_method=='sum':
            self.qk_compact = self._qk_compact_v1          
        elif correlation_method=='mean':
            self.qk_compact = self._qk_compact_v2
        elif correlation_method=='linear':
            # Add normalization here? 
            self.shrink_2_1 = torch.nn.Linear(in_features=TimeDim*scales[0]*scales[1]*scales[2],out_features=1)
            self.qk_compact = self._qk_compact_v3   
        else:
            raise ValueError("Cannot understand correlation method, aborting ...")
    

 

    # These are numerically stable versions 
    def _qk_identity_sim_v1(self,q,k,smooth=1.e-5): 
        # q --> B, T, c, h, w, C/c, H/h, W/w 
        # k --> B, T, c, h, w, C/c, H/h, W/w   
        #print (q.shape)

        scale = np.reciprocal(np.sqrt(np.prod(q.shape[-3:])))
        q = q*scale
        k = k*scale

        qk = einsum('iWjklmno,iPstrmno->iWjklPstr',q,k) #B, T x S x c, h, w, c, h, w
        qq = einsum('iWjklmno,iWjklmno->iWjkl',q,q) #B, T x S , c, h, w  
        kk = einsum('iPstrmno,iPstrmno->iPstr',k,k) #B, T x S , c, h, w


        #print("num minmax:: ",qk.min(),qk.max())

        denum = (qq[:,:,:,:,:,None,None,None,None]+kk[:,None,None,None,None])-qk + smooth
        #print("denum minmax:: ",denum.min(),denum.max())

        logqk = torch.log(qk+smooth)
        logdenum = torch.log(denum)
        result = torch.exp(logqk - logdenum)

        return result

    def _qk_identity_sim_v2(self,q,k,smooth=1.e-5): 
        # This version scales down the gradients for depth > 5.0, recommend depth=10.0
        # q --> B, T, c, h, w, C/c, H/h, W/w 
        # k --> B, T, c, h, w, C/c, H/h, W/w   
        #print (q.shape)

        scale = np.reciprocal(np.sqrt(np.prod(q.shape[-3:])))
        q = q*scale
        k = k*scale

        qk = einsum('iWjklmno,iPstrmno->iWjklPstr',q,k) #B, T x S x c, h, w, c, h, w
        qq = einsum('iWjklmno,iWjklmno->iWjkl',q,q) #B, T x S , c, h, w  
        kk = einsum('iPstrmno,iPstrmno->iPstr',k,k) #B, T x S , c, h, w

        denum = self.alpha*(qq[:,:,:,:,:,None,None,None,None]+kk[:,None,None,None,None])-self.beta*qk + smooth
        # Avoid overflow problems 
        logqk = torch.log(qk+smooth)
        logdenum = torch.log(denum)
        result = torch.exp(logqk - logdenum)


        return result




    def _qk_compact_v1(self,qk):
        # input qkv: B x S x (T*c*h*w) x [c x h x w] x C//c x H//h x W//w 
        # output v : B x S x C x H x W 
        tqk = torch.sum(qk,dim=1)
        return tqk 
        
    def _qk_compact_v2(self,qk):
        # input qkv: B x S x (T*c*h*w) x [c x h x w] x C//c x H//h x W//w 
        # output v : B x S x C x H x W 
        tqk = torch.mean(qk,dim=1)
        return tqk 
        
        
    def _qk_compact_v3(self,qk):
        # input qkv: B x S x (T*c*h*w) x [c x h x w] x C//c x H//h x W//w 
        # output v : B x S x C x H x W 
        tqk = qk.permute(0,2,3,4,5,1)
        tqk2 = self.shrink_2_1(tqk).squeeze(dim=-1)
        
        return tqk2
   
                 
    
    def qk_select_v(self,qk,vpatch, smooth=1.e-5):
        # @@@@@@@@@@@@@ MAYDAY: Provide various ways of summing up weights 
        
        # qk --> B, c, h, w, c, h, w 
        # v  -->B , c, h, w, C/c, H/h, W/w 
        # qk: similarity between q and k values 
        # v : values 
        # qkv: v values emphasized where qk says it's important 
        
        # Options 1, 2 and 3 unified in this method 
        #print(qk.shape)
        tqk = qk.reshape(qk.shape[0],-1,*qk.shape[5:]) # B x (T*c*h*w) x [S x c x h x w]


        #print(tqk.shape)
        tqk =  self.qk_compact(tqk) # B x [c x h x w]
        qkvv = einsum('bTrst, bTrstmno -> bTrstmno', tqk,vpatch)
        
        
        qkvv = self.patchify._2tensor(qkvv)
        return qkvv

    
    def get_att(self, q,k,v):
        # =================================================================================        
        qp = self.patchify._2patch(q) # B, T, c, h, w, C//c, H//h, W//w  # <==================================================
        kp = self.patchify._2patch(k) # B, T, c, h, w, C//c, H//h, W//w  # <==================================================
        vp = self.patchify._2patch(v) # B, T, c, h, w, C//c, H//h, W//w  # <================================================== 
        #print ("pass1 ")

        #print(q.shape)
        #print(qp.shape)
        
        qpkp = self.qk_sim(qp,kp) # B x T x (c x h x w) x [c x h x w]  # ORIGINAL 

        vout    = self.qk_select_v(qpkp,vp)  # B x T x C x H x W 

        return vout   






class RelPatchAttention3DTCHW(torch.nn.Module):
    # Fastest implementation so far  - with sigmoid 
    def __init__(self,in_channels,out_channels,scales,kernel_size=(3,3,3),padding=(1,1,1),nheads=1,norm='BatchNorm',norm_groups=None, correlation_method='sum',  TimeDim=None,depth=0.0):
        super().__init__()
       


        self.act =   D2Sigmoid(scale=False)
        self.patch_attention = BASE_RelPatchAttention3D_TCHW(out_channels, scales, correlation_method=correlation_method,TimeDim=TimeDim,depth=depth)

        self.query   = Conv3DNormed(in_channels=in_channels,out_channels=out_channels,kernel_size= kernel_size, padding = padding, norm_type= norm, num_groups=norm_groups, groups=nheads)
        self.kv      = Conv3DNormed(in_channels=in_channels,out_channels=out_channels*2,kernel_size= kernel_size, padding = padding, norm_type= norm, num_groups=norm_groups, groups=nheads*2)

        
    def forward(self,input1:torch.Tensor, input2:torch.Tensor):

        # CONFIGURATION AVOIDS NANs
        # Best configuration so far to avoid nans 
        # @@@@@@@@@@@@@ SUCCESSS @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # MAYDAY: Changing sigmoid activation to d2sigmoid solved the problem of nans, but added the problem of LARGE memory footprint 
        # I need to remove sigmoid as much as possible, and re-think of that 
        q    = self.query(input1) # B,C,H,W
        #k    = self.key(input2) # B,C,H,W
        #v    = self.value(input2) # B,C,H,W
        k,v  = self.kv(input2).split(q.shape[1],1)
        
        #print (q.shape, k.shape, v.shape)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@ NEW  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 
        # seems to be helping with the nans which are probably due to too much zeros after similarity 
        # This is scaling of the inputs basically AND adding non linearity for final Linear 
        # best avoiding of nan behaviour so far 
        q    = self.act(q)+0.1 # This helps with log transformations inside the qksim
        k    = self.act(k)+0.1 # This helps with log transformations inside the qksim
        # @@@@@@@@@@@@@@@@@@@@@@@@ NEW @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ 
        q = q.permute(0,2,1,3,4)
        k = k.permute(0,2,1,3,4)
        v = v.permute(0,2,1,3,4)
        v    = self.patch_attention.get_att(q,k,v)
        v    = v.permute(0,2,1,3,4)

        v    = self.act(v) 
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        return v


