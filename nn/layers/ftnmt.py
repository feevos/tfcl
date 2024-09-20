import torch
from typing import List

__all__=['FTanimoto']

def inner_prod(prob, label,axis:List[int]):
    return (prob * label).sum(dim=axis,keepdim=True)


# Custom definition that avoids resulting in exploding gradients when both inputs are zero.
class tnmt_2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx,p:torch.Tensor,l:torch.Tensor,d:int,axis:List[int]=[2,3]): 
        pl = inner_prod(p,l,axis)
        pp = inner_prod(p,p,axis)
        ll = inner_prod(l,l,axis)

        
        a = 2**d
        b = -(2.*a-1.)
        
        denum = a*(pp+ll) + b*pl #+ 1.e-5
        scale = torch.reciprocal(denum)
        scale = torch.nan_to_num(scale, nan=0.0,posinf=1.,neginf=-1)
        
        ctx.save_for_backward(p,l,pl,pp,ll,scale)
        ctx.a = a
        
        result = pl*scale
        
        return result
    
    @staticmethod 
    def backward(ctx,grad_output):
        # grad_output is the derivative with respect to some loss, assume summation
        p, l, pl, pp, ll, scale = ctx.saved_tensors
        a = ctx.a

        #ascale2 = a*scale**2
        ascale2 = (a*scale)*scale
        ppmll  = pp+ll
        
        result_p = ascale2 *(-2.*p*pl + l *ppmll)
        result_l = ascale2 *(-2.*l*pl + p *ppmll)
                
        return result_p  * grad_output, result_l  * grad_output, None, None     


class FTanimoto(torch.nn.Module):
    """
    This is the average fractal Tanimoto set similarity with complement.
    """
    def __init__(self, depth=0, axis=[2,3],mode='exact'):
        super().__init__()

        if depth == 0:
            self.scale=1.
        else:
            self.scale = 1./(depth+1.)

        self.depth=depth
        self.axis=axis

        if mode=='exact' or depth==0:
            self.tnmt_base = self.tnmt_base_exact
        elif mode=='avg':
            self.tnmt_base = self.tnmt_base_avg
        else:
            raise  ValueError("variable mode must be one of 'avg' or 'exact', default == 'avg'")

    def set_depth(self,depth):
        assert depth >= 0, "Expecting depth >= 0, aborting ..."
        if depth == 0:
            scale=1.
        else:
            scale = 1./(depth+1.)

        self.scale = torch.tensor(scale)
        self.depth = depth 

    @torch.jit.export
    def tnmt_base_avg(self, preds, labels):
        if self.depth==0:
            return tnmt_2d.apply(preds,labels,self.depth,self.axis)
        else:
            result = 0.0
            for d in range(self.depth+1):
                result = result + tnmt_2d.apply(preds,labels,d,self.axis)

            return result * self.scale


    @torch.jit.export
    def tnmt_base_exact(self, preds, labels):
        return tnmt_2d.apply(preds,labels,self.depth,self.axis)

    def forward(self, preds, labels):
            l12 = self.tnmt_base(preds,labels)
            l12 = l12 + self.tnmt_base(1.-preds, 1.-labels)

            return 0.5*l12






