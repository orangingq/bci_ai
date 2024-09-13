
import argparse
import torch
from torchvision import models as torch_models

seed = 42
dataset = 'BCI_dataset'
image_size = 224
model_name = 'ViT'
optimizer_name = 'Adam'
learning_rate = 1e-3
log_freq = 10
load_dir = None # default : pretrained model 
save_dir = None # default : not save

def get_args():
    '''Get Global Variables as Dictionary'''
    global_variables = [
        'seed', 
        'dataset', 'image_size',
        'model_name',
        'optimizer_name', 'learning_rate', 'log_freq',
        'load_dir', 'save_dir'
    ]
    return {var: globals()[var] for var in global_variables}

def set_args():
    '''Set Arguments from Command Line'''
    global seed, dataset, image_size, model_name, optimizer_name, learning_rate, log_freq, load_dir, save_dir
    
    parser = argparse.ArgumentParser()
    # Random seed
    parser.add_argument('--seed', type=int, default=seed, help='Random Seed')
    # dataset arguments
    parser.add_argument('--dataset', type=str, default=dataset, help='Dataset')
    parser.add_argument('--image_size', type=int, default=image_size, help='Image Size')
    # model arguments
    parser.add_argument('--model_name', type=str, default=model_name, help='Model Name')
    # optimizer arguments
    parser.add_argument('--optimizer_name', type=str, default=optimizer_name, help='Optimizer')
    parser.add_argument('--learning_rate', type=float, default=learning_rate, help='Learning Rate')
    # log arguments
    parser.add_argument('--log_freq', type=int, default=log_freq, help='Log Frequency')
    # load/save arguments
    parser.add_argument('--load_dir', type=str, default=load_dir, help='Load Directory')
    parser.add_argument('--save_dir', type=str, default=save_dir, help='Save Directory')

    args = parser.parse_args()
    for arg in vars(args):
        globals()[arg] = getattr(args, arg)# update global variables
    return


def get_model():
    '''Return Model from Model Name (args.model_name)'''
    if model_name.lower() == 'vit':
        model = torch_models.vit_b_16(weights=torch_models.ViT_B_16_Weights.IMAGENET1K_V1)
    elif model_name.lower() in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        model_size = model_name.split('resnet')[-1]
        model = torch_models.__dict__[f'resnet{model_size}'](weights=torch_models.__dict__[f'ResNet{model_size}_Weights'].IMAGENET1K_V1)
    else:
        raise ValueError(f'Model name {model_name} is not available')
    return model

def get_optimizer(model: torch.nn.Module):
    '''Return Optimizer from Optimizer Name (args.optimizer)'''
    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f'Optimizer {optimizer} is not available')
    return optimizer