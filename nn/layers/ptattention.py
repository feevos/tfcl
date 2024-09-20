import torch
from torch import einsum
from tfcl.nn.layers.conv2Dnormed import Conv2DNormed
from tfcl.utils.get_norm import * 


class PatchifyCHW(torch.nn.Module):
    def __init__(self, cscale, hscale, wscale):
        super().__init__()
        self.c = cscale
        self.h = hscale
        self.w = wscale
        self.unfold_shape = None

    def _2patch(self,input):
        shape = input.shape
        c     = torch.div(shape[1], self.c, rounding_mode='floor')
        h     = torch.div(shape[2], self.h, rounding_mode='floor')
        w     = torch.div(shape[3], self.w, rounding_mode='floor')

        # Currently only works with stride = window size
        sc    = c  
        sh    = h 
        sw    = w

        # Here I assume stride is equal to c
        patch = input.unfold(1,c,sc).unfold(2,h,sh).unfold(3,w,sw)
        self.unfold_shape = patch.shape
        return patch

    def _2tensor(self, patch):
        output_c    = self.unfold_shape[1] * self.unfold_shape[4]
        output_h    = self.unfold_shape[2] * self.unfold_shape[5]
        output_w    = self.unfold_shape[3] * self.unfold_shape[6]
        tensorpatch = patch.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        tensorpatch = tensorpatch.view(self.unfold_shape[0],output_c,output_h,output_w)
        return tensorpatch



class BASE_RelPatchAttention2D_v2(torch.nn.Module):
    def __init__(self, nfilters, spatial_size,  scales, metric_learning=True, correlation_method='linear'):
        super().__init__()                                                                                                                                                                                                                                                                                
                       
        self.scales = scales
        self.patchify = PatchifyCHW(cscale=scales[0],hscale=scales[1],wscale=scales[1])                                                                                                                                                                                                 
        
        
        self.qk_sim = self._qk_identity_sim
        
        if correlation_method=='sum':
            self.qk_compact = self._qk_compact_v1          
        elif correlation_method=='mean':
            self.qk_compact = self._qk_compact_v2
        elif correlation_method=='linear':
            self.shrink_2_1 = torch.nn.Linear(in_features=scales[0]*scales[1]**2,out_features=1)
            self.qk_compact = self._qk_compact_v3
        
        else:
            raise ValueError("Cannot understand correlation method, aborting ...")
        

    def _qk_identity_sim(self,q,k,smooth=1.e-5): 
        # q --> B, c, h, w, C/c, H/h, W/w 
        # k --> B, c, h, w, C/c, H/h, W/w                                                                                                                                                                                                                                                                                          

        qk = einsum('ijklmno,istrmno->ijklstr',q,k) #B, c, h, w, c, h, w
        qq = einsum('ijklmno,ijklmno->ijkl',q,q) #B, c, h, w  
        kk = einsum('istrmno,istrmno->istr',k,k) #B, c, h, w
 

        denum = (qq[:,:,:,:,None,None,None]+kk[:,None,None,None])-qk +smooth
        result = (qk+smooth) * denum.reciprocal()
 
        return result

    
    def _qk_compact_v1(self,qk):
        # input qkv: B x (c*h*w) x [c x h x w] x C//c x H//h x W//w 
        # output v : B x C x H x W 
        tqk = torch.sum(qk,dim=1)
        return tqk 
        
    def _qk_compact_v2(self,qk):
        # input qkv: B x (c*h*w) x [c x h x w] x C//c x H//h x W//w 
        # output v : B x C x H x W 
        tqk = torch.mean(qk,dim=1)
        return tqk 
        
        
    def _qk_compact_v3(self,qk):
        # input qkv: B x (c*h*w) x [c x h x w] 
        # output v : B x [c x h x w] 
        tqk = qk.permute(0,2,3,4,1)
        tqk2 = self.shrink_2_1(tqk).squeeze(dim=-1)
        
        return tqk2
   
                 
    
    def qk_select_v(self,qk,vpatch, smooth=1.e-5):
        # qk      --> B, c, h, w, c  , h  , w 
        # vpatch  --> B, c, h, w, C/c, H/h, W/w 
        # qk: similarity between q and k values 
        # vpatch : values patchified
        # qkv: v values emphasized where qk says it's important. Replaces softmax selection 
        
        # Options 1, 2 and 3 unified in this method 
        tqk = qk.reshape(qk.shape[0],-1,*qk.shape[4:]) # B x (c*h*w) x [ c x h x w]
        tqk =  self.qk_compact(tqk) # B x [c x h x w]
        # ELEMENT WISE multiplication
        qkv = einsum('brst, brstmno -> brstmno', tqk,vpatch) # B x [c x h x w] x C/c x H/h x W/w
        
        
        qkv = self.patchify._2tensor(qkv)
        return qkv

    
    def get_att(self, q,k,v):
        # =================================================================================        
        qp = self.patchify._2patch(q) # B, (c, h, w), C//c, H//h, W//w  
        kp = self.patchify._2patch(k) # B, [c, h, w], C//c, H//h, W//w  
        vp = self.patchify._2patch(v) # B, [c, h, w], C//c, H//h, W//w  

        # Note, ( ) indices are QUERY indices, [ ] indices are KEY-VALUE indices
        qpkp = self.qk_sim(qp,kp) # B x (c x h x w) x [c x h x w]  

        vout    = self.qk_select_v(qpkp,vp)  # B x C x H x W 

        return vout   






from tfcl.nn.activations.d2sigmoid import * 
class RelPatchAttention2D(torch.nn.Module):
    # Fastest implementation so far  - with sigmoid 
    def __init__(self,in_channels,out_channels,spatial_size,scales,kernel_size=3,padding=1,nheads=1,norm='BatchNorm',
                 norm_groups=None, correlation_method='linear'):
        super().__init__()
       
        self.act =   D2Sigmoid(scale=False)
        self.patch_attention = BASE_RelPatchAttention2D_v2(out_channels, spatial_size,  scales, correlation_method=correlation_method)
        self.query   = Conv2DNormed(in_channels=in_channels,out_channels=out_channels,kernel_size= kernel_size, padding = padding, norm_type= norm, num_groups=norm_groups, groups=nheads)
        self.kv      = Conv2DNormed(in_channels=in_channels,out_channels=out_channels*2,kernel_size= kernel_size, padding = padding, norm_type= norm, num_groups=norm_groups, groups=nheads*2)


    def forward(self,input1:torch.Tensor, input2:torch.Tensor):

        q    = self.query(input1) # B,C,H,W
        k,v  = self.kv(input2).split(q.shape[1],1) # B,C,H,W each 
        
        q    = self.act(q)
        k    = self.act(k)

        v    = self.patch_attention.get_att(q,k,v)
        v    = self.act(v) 

        return v


class PatchAttention2D(torch.nn.Module):
    def __init__(self,in_channels, out_channels, spatial_size, scales=None, kernel_size=3, padding=1,nheads=1, norm = 'BatchNorm', norm_groups=None, correlation_method='linear'):
        super().__init__()

        self. att = RelPatchAttention2D(in_channels=in_channels,
                                        out_channels = out_channels,
                                        spatial_size=spatial_size, 
                                        scales=scales,
                                        kernel_size=kernel_size,
                                        padding=padding,
                                        nheads=nheads,
                                        norm = norm,
                                        norm_groups=norm_groups,
                                        correlation_method=correlation_method
                                        )


    def forward(self, input:torch.Tensor):
        return self.att(input,input)



