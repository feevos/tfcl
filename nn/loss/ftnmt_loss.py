import torch
from tfcl.nn.layers.ftnmt import * 


class ftnmt_loss(torch.nn.Module):
    def __init__(self, depth=0, axis=[2,3], mode='exact'):
        super(ftnmt_loss,self).__init__()

        self.ftnmt = FTanimoto(depth=depth, axis=axis,mode=mode)


    def forward(self,preds,labels):
        sim = self.ftnmt(preds,labels)

        return (1. - sim).mean()
