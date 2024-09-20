import torch

from tfcl.nn.layers.scale import *
from tfcl.nn.activations.sigmoid_crisp import *
from tfcl.nn.layers.conv2Dnormed import *


######__all__ = ['head_cmtsk','head_cmtsk_lstm']


# Helper classification head, for a single layer output 
class HeadSingle(torch.nn.Module):
    def __init__(self, in_channels, out_channels,  NClasses, depth=2, norm_type='BatchNorm',norm_groups=None, **kwargs):
        super().__init__(**kwargs)

        logits = [] 
        logits.append( Conv2DNormed(in_channels = in_channels, out_channels = out_channels, kernel_size = (3,3),padding=(1,1), norm_type=norm_type, num_groups=norm_groups))
        for _ in range(depth-1):
            logits.append( Conv2DNormed(in_channels = out_channels, out_channels = out_channels,kernel_size = (3,3),padding=(1,1), norm_type=norm_type, num_groups=norm_groups))
            logits.append( torch.nn.ReLU())

        # This was a bug, living inside the for loop    
        logits.append( torch.nn.Conv2d(in_channels=out_channels, out_channels=NClasses,kernel_size=1,padding=0))
        self.logits = torch.nn.Sequential(*logits)

    def forward(self,input):
        return self.logits(input)






class head_cmtsk(torch.nn.Module):
    def __init__(self, nfilters, NClasses, nfilters_embed=32, spatial_size=256,scales=(4,8),   norm_type = 'BatchNorm', norm_groups=None,segm_act ='softmax'):
        super().__init__()
        
        self.model_name = "Head_CMTSK_BC" 

        self.nfilters = nfilters_embed # Initial number of filters 
        self.NClasses = NClasses

        # distance logits -- deeper for better reconstruction 
        self.distance_logits = HeadSingle(in_channels = nfilters_embed, out_channels = nfilters_embed,  NClasses = NClasses, norm_type = norm_type, norm_groups=norm_groups)
        self.dist_Equalizer = Conv2DNormed(in_channels = NClasses, out_channels = self.nfilters,kernel_size =1, norm_type=norm_type, num_groups=norm_groups)


        self.Comb_bound_dist =  Conv2DNormed(in_channels= nfilters_embed*2, out_channels = self.nfilters,kernel_size =1, norm_type=norm_type, num_groups=norm_groups)

       
        # bound logits 
        self.bound_logits = HeadSingle(in_channels = nfilters_embed*2, out_channels = nfilters_embed,NClasses=NClasses, norm_type = norm_type, norm_groups=norm_groups)
        self.bound_Equalizer = Conv2DNormed(in_channels=NClasses, out_channels = self.nfilters,kernel_size =1, norm_type=norm_type, num_groups=norm_groups)
            

        # Segmenetation logits -- deeper for better reconstruction 
        self.final_segm_logits = HeadSingle(in_channels = nfilters_embed*2, out_channels=nfilters_embed, NClasses = NClasses, norm_type = norm_type, norm_groups=norm_groups)
         
        self.CrispSigm = SigmoidCrisp()

        # Last activation, customization for binary results
        if ( self.NClasses == 1):
            self.ChannelAct = SigmoidCrisp() #-> torch.nn.Module
            self.segm_act   = SigmoidCrisp() 
        else:
            #TODO add scaled softmax as well
            if segm_act =='softmax':
                self.segm_act   = torch.nn.Softmax(dim=1)  
            elif segm_act =='sigmoid':
                self.segm_act   = SigmoidCrisp() 
            else:
                raise ValueError("I don't understand type of segm_act, aborting ...")
            self.ChannelAct = torch.nn.Softmax(dim=1) 



    def forward(self, conv):
        # logits 

        # 1st find distance map, skeleton like, topology info
        dist = self.distance_logits(conv) # do not use max pooling for distance
        dist = self.ChannelAct(dist)
        distEq = torch.relu(self.dist_Equalizer(dist)) # makes nfilters equals to conv 


        # Then find boundaries 
        bound = torch.cat([conv, distEq],dim=1)
        bound = self.bound_logits(bound)
        bound   = self.CrispSigm(bound) # Boundaries are not mutually exclusive 
        boundEq = torch.relu(self.bound_Equalizer(bound))


        # Now combine all predictions in a final segmentation mask 
        # Balance first boundary and distance transform, with the features
        comb_bd = self.Comb_bound_dist(torch.cat([boundEq, distEq],dim=1))
        comb_bd = torch.relu(comb_bd)

        all_layers = torch.cat([comb_bd, conv],dim=1)
        final_segm = self.final_segm_logits(all_layers)
        final_segm = self.segm_act(final_segm)


        return  torch.cat([final_segm, bound, dist],dim=1)





