import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import  ImageOnlyTransform
class TrainingTransformS2(object):                                                                                                                                
    # Built on Albumentations, this provides geometric transformation only  
    def __init__(self,  prob = 1., mode='train', compress=255.):                                                                              
        self.distance_scale=1. / compress # This is the scaling of the mask distance transform                                                                      
        self.geom_trans = A.Compose([   
                    A.OneOf([          
                        A.HorizontalFlip(p=1),   
                        A.VerticalFlip(p=1),    
                        A.ElasticTransform(p=1), # VERY GOOD - gives perspective projection, really nice and useful - VERY SLOW   
                        A.GridDistortion(distort_limit=0.4,p=1.),   
                        A.ShiftScaleRotate(shift_limit=0.25, scale_limit=(0.75,1.25), rotate_limit=180, p=1.0), # Most important Augmentation   
                        ],p=1.) 
                    ],         
            additional_targets={'imageS1': 'image','mask':'mask'}, 
            p = prob)                                             
        if mode=='train':   
            self.mytransform = self.transform_train   
        elif mode =='valid':   
            self.mytransform = self.transform_valid  
        else:   
            raise ValueError('transform mode can only be train or valid') 
    def transform_valid(self, data):  
        timgS2, tmask = data         
        tmask= tmask * self.distance_scale  
        return timgS2,  tmask.astype(np.float32)                                                                                                             
    def transform_train(self, data):   
        timgS2, tmask = data          
        tmask= tmask * self.distance_scale   
        tmask = tmask.astype(np.float32)    
        # Special treatment of time series
        c2,t,h,w = timgS2.shape          
        #print (c2,t,h,w)              
        timgS2 = timgS2.reshape(c2*t,h,w)  
        result = self.geom_trans(image=timgS2.transpose([1,2,0]),
                                 mask=tmask.transpose([1,2,0])) 
        timgS2_t = result['image']    
        tmask_t  = result['mask']    
        timgS2_t = timgS2_t.transpose([2,0,1])  
        tmask_t = tmask_t.transpose([2,0,1])   
        timgS2_t = timgS2_t.reshape(c2,t,h,w)  
        return timgS2_t,  tmask_t                                                                                                                          
    def __call__(self, *data):      
        return self.mytransform(data)                                                                                                                               
                                                   
