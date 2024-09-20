import torch


def get_norm3d(name,channels,num_groups=None):
    if name == 'BatchNorm':
        return torch.nn.BatchNorm3d(num_features=channels)
    elif name == 'GroupNorm':
        return torch.nn.GroupNorm(num_channels=channels,num_groups=num_groups)
    else:
        raise ValueError("I do not understand normalization name::{}, options:: BatchNorm, GroupNorm, aborting ...".format(name))

def get_norm2d(name,channels,num_groups=None):
    if name == 'BatchNorm':
        return torch.nn.BatchNorm2d(num_features=channels)
    elif name == 'GroupNorm':
        return torch.nn.GroupNorm(num_channels=channels,num_groups=num_groups)
    else:
        raise ValueError("I do not understand normalization name::{}, options:: BatchNorm, GroupNorm, aborting ...".format(name))



def get_norm1d(name,channels,num_groups=None):
    if name == 'BatchNorm':
        return torch.nn.BatchNorm1d(num_features=channels)
    elif name == 'GroupNorm':
        return torch.nn.GroupNorm(num_channels=channels,num_groups=num_groups)
    else:
        raise ValueError("I do not understand normalization name::{}, options:: BatchNorm, GroupNorm, aborting ...".format(name))

