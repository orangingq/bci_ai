
import argparse
import torch
import torch.nn as nn
from torchvision import models as torch_models

seed = 42
dataset = 'BCI_dataset'
aug_level = 0 # 0: no augmentation, 1: simple augmentation (rotation), 2: complex augmentation
image_size = 224
model_name = 'ViT'
optimizer_name = 'Adam'
learning_rate = 1e-3
log_freq = 30
finetune = False
load_dir = None # default : load pretrained model and fine-tune
save_dir = None # default : not save
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_args():
    '''Get Global Variables as Dictionary'''
    global_variables = [
        'seed', 
        'dataset', 'aug_level', 'image_size',
        'model_name',
        'optimizer_name', 'learning_rate', 'log_freq', 'finetune',
        'load_dir', 'save_dir', 'device'
    ]
    return {var: globals()[var] for var in global_variables}

def set_args():
    '''Set Arguments from Command Line'''
    global seed, dataset, aug_level, image_size, model_name, optimizer_name, learning_rate, log_freq, finetune, load_dir, save_dir, device
    
    parser = argparse.ArgumentParser()
    # Random seed
    parser.add_argument('--seed', type=int, default=seed, help='Random Seed')
    # dataset arguments
    parser.add_argument('--dataset', type=str, default=dataset, help='Dataset')
    parser.add_argument('--aug_level', type=int, default=aug_level, help='Augmentation Level')
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
    parser.add_argument('--finetune', action='store_true', help='Fine-tune the classification model')

    args = parser.parse_args()
    for arg in vars(args):
        globals()[arg] = getattr(args, arg)# update global variables
    print(f'Arguments are set as follows: {get_args()}')
    return

def get_dataloader(type:str='classification', batch_size:int=32, num_workers:int=40, image_size=image_size, aug_level=aug_level)->dict:
    '''Return Dataloaders from Dataset Name (args.dataset)'''
    if dataset.lower() == 'acrobat':
        from datasets.acrobat.dataloader import get_acrobat_dataloaders
        dataloaders = get_acrobat_dataloaders(type=type, batch_size=batch_size, num_workers=num_workers, image_size=image_size, aug_level=aug_level)
    elif dataset.lower() == 'bci_dataset':
        from datasets.BCI_dataset.dataloader import get_bci_dataloaders
        dataloaders = get_bci_dataloaders(type=type, batch_size=batch_size, num_workers=num_workers, image_size=image_size, aug_level=aug_level)
    else:
        raise ValueError(f'Dataset {dataset} is not available')
    return dataloaders

def get_model(num_classes:int)->nn.Module:
    '''Return Models (Classification model & Segmentation model) from Model Name (args.model_name)'''    
    if model_name.lower() == 'vit':
        model = torch_models.vit_b_16(weights=torch_models.ViT_B_16_Weights.IMAGENET1K_V1)
        fc_layer = model.heads.head # last layer
    elif model_name.lower() in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
        model_size = model_name.split('resnet')[-1]
        model = torch_models.__dict__[f'resnet{model_size}'](weights=torch_models.__dict__[f'ResNet{model_size}_Weights'].IMAGENET1K_V1)
        fc_layer = model.fc # last layer
    else:
        raise ValueError(f'Model name {model_name} is not available')
    
    # Change num_classes of the Last Layer 
    fc_layer.out_features = num_classes
    fc_layer.weight.data = fc_layer.weight[:num_classes, :] # HER2 levels
    fc_layer.bias.data = fc_layer.bias[:num_classes] # HER2 levels
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