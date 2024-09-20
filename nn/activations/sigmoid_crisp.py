import torch

class SigmoidCrisp(torch.nn.Module):
    # Tempered sigmoid activation, from Diakogiannis et al 2021 https://www.mdpi.com/2072-4292/13/18/3707 
    def __init__(self,smooth=1.e-2):
        super(SigmoidCrisp,self).__init__()


        self.smooth = smooth
        self.gamma = torch.nn.Parameter(torch.ones(1),requires_grad=True)

    def forward(self,input):
        # This guarantees that out > 0.0 
        out = self.smooth + torch.sigmoid(self.gamma)
        out = torch.reciprocal(out)

        out = input*out 
        out = torch.sigmoid(out)

        return out 
