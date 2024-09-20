import torch

__all__=['D2Sigmoid']

class D2SigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        
        u = torch.special.expit(-x)
        ctx.save_for_backward(u)
        return u*(1. + u*(-3. + 2.*u))
        
    @staticmethod
    def backward(ctx,grad_output):
        u = ctx.saved_tensors[0]
        return u*(-1. + u*(7. + u*(-12. + 6.*u)))*grad_output

class D2Sigmoid(torch.nn.Module):
    def __init__(self,scale=False):
        super(D2Sigmoid,self).__init__()
        
    def forward(self,input):
        return D2SigmoidFunction.apply(input)


