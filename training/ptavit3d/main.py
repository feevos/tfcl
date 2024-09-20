# Based on multiprocessing example from
# https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
import sys
sys.path.append(r'../../../../') # location of tfc directory relative to main file


from datetime import datetime
import argparse
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from tfcl.models.ptavit.ptavit3d_dn import ptavit3d_dn
from tfcl.nn.loss.ftnmt_loss import ftnmt_loss
from tfcl.utils.classification_metric import Classification 
from tfcl.data.transform import *
from tfcl.data.rocksdbutils import * 

# Debugging flag - set to False for training.
DEBUG=True 

# Here NClasses = 6 for ISPRS
def mtsk_loss(preds, labels,criterion, NClasses):
    # Multitasking loss,    segmentation / boundaries/ distance

    pred_segm  = preds[:,:NClasses]                                                                                           
    pred_bound = preds[:,NClasses:2*NClasses]                                                                            
    pred_dists = preds[:,2*NClasses:3*NClasses]                                                                          
                                                                                                                               
                                                                                                                               
                                                                                                                               
    # Multitasking loss                                                                                                            
    label_segm  = labels[:,:NClasses]                                                                                         
    label_bound = labels[:,NClasses:2*NClasses]                                                                          
    label_dists = labels[:,2*NClasses:3*NClasses]                                                                        
                                                                                                                               
                                                                                                                               
    loss_segm  = criterion(pred_segm,   label_segm)                                                   
    loss_bound = criterion(pred_bound, label_bound) 
    loss_dists = criterion(pred_dists, label_dists) 
                                                                                                                               
                                                                                                                               
    return (loss_segm+loss_bound+loss_dists)/3.0                                                                                   



def monitor_epoch(model, epoch, datagen_valid, NClasses):
    # Computes various classification metrics                                                                                                                  
    metric_target   = Classification(num_classes=NClasses).cuda() #                                        
                                                                                                                                         
                                                                                                                                         
    dist.barrier() # Make sure all operations finish until this point                                                                        
                                                                                                                                         
    for idx, data in enumerate(datagen_valid): 
        images,labels = data

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True) 

        # preds_target is the output of the sequence model 
        with torch.inference_mode():
            preds_target= model(images) 


        # XXXXXXXXXXXXXXXXXXXXX TARGET metrics XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX                                      
        pred_segm  = preds_target[:,:NClasses]  # Segmentation only output                                       
        label_segm = labels[:,:NClasses]                                                                


        # Update Metric                                                                                               
        metric_target(pred_segm,torch.argmax(label_segm,dim=1) )                                                   

        # DEBUGGING OPTION
        if DEBUG and idx > 5: 
            break 

    # Evaluate statistics for all predictions 
    metric_kwargs_target = metric_target.compute()

    kwargs = {'epoch':epoch}
    for k,v in metric_kwargs_target.items():                               
        kwargs[k+"_target_vV"]=v.cpu().numpy() # Pass to cpu and numpy format 


    return kwargs



def train(args):
    num_epochs = args.epochs
    batch_size = args.batch_size


    dist.init_process_group(backend='nccl')

    torch.manual_seed(0)
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)


    NClasses=6 # 
    nf=96
    verbose = dist.get_rank() == 0  # print only on global_rank==0 
    model_config = {'in_channels':5,
                   'spatial_size_init':(128,128),         
                   'depths':[2,2,5,2],              
                   'nfilters_init':nf,              
                   'nheads_start':nf//4,            
                   'NClasses':NClasses,   
                   'verbose':verbose,               
                   'segm_act':'sigmoid'}                  
 
    # UNet-like model 
    model = ptavit_dn(**model_config).cuda()

    # Fractal Tanimoto with complement loss 
    criterion = ftnmt_loss()
    criterion_features = ftnmt_loss(axis=[-3,-2,-1])
    
    # You might need eps = 1.e-6 if you train with mixed precision. It avoids NANs due to overflow. 
    optimizer = torch.optim.RAdam(model.parameters(), lr=1e-3, eps=1.e-6)

    model = DistributedDataParallel(model, device_ids=[local_rank])


    transform_train = TrainingTransform(NClasses=NClasses,mode='train')
    transform_valid = TrainingTransform(NClasses=NClasses,mode='valid')

    train_dataset = RocksDBDataset(                                                                                                    
                 flname_db= '../../data/train.db', # Change here with location of your database
                 transform=transform_train,
                 num_workers=4) # 4 is a good choice for RocksDB                                       
                                                                                                                                                      


    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True,
                              sampler=train_sampler)



    valid_dataset = RocksDBDataset(                                                                                                    
                 flname_db= '../../data/valid.db', # Change here with location of your database
                 transform=transform_valid,
                 num_workers=4) # 4 is a good choice for RocksDB                                       
                                                                                                                                                      


    valid_sampler = DistributedSampler(valid_dataset)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True,
                              sampler=valid_sampler)


    start = datetime.now()
    for epoch in range(num_epochs):
        tot_loss = 0
        for i, data in enumerate(train_loader):

            # #######################################
            # DEBUGGING OPTION
            if DEBUG and  i > 5 : 
                break 
            # ########################################

            images,labels = data

            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)


            optimizer.zero_grad(set_to_none=True)   
            preds_target = model(images)    
                                                          
            # Target loss -- Predictions of sequence model 
            loss = mtsk_loss(preds_target, labels,criterion,NClasses)
            loss.backward()
            optimizer.step()

            tot_loss += loss.item()

        kwargs = monitor_epoch(model, epoch, valid_loader, NClasses)
        kwargs['tot_train_loss'] = tot_loss
        if verbose:
            output_str = ', '.join(f'{k}:: {v}, |===|, ' for k, v in kwargs.items())
            print(output_str)

    if verbose:
        print("Training completed in: " + str(datetime.now() - start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=2, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch-size', default=2, type=int, metavar='batch', help='batch-size')
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()

