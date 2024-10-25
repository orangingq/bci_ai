import glob
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from tqdm import tqdm
import random
import torchvision.transforms.v2 as v2
from utils import path 

data_dir = path.get_data_dir('acrobat')

# https://github.com/adamtupper/medical-image-augmentation/blob/main/individual_effects.py
AUGMENTATIONS = {
    # "center_crop": [v2.CenterCrop(224)],
    # "random_crop": [v2.RandomCrop(224)],
    "horizontal_flip": [v2.CenterCrop(224), v2.RandomHorizontalFlip()],
    "vertical_flip": [v2.CenterCrop(224), v2.RandomVerticalFlip()],
    "rotation": [v2.CenterCrop(224), v2.RandomRotation(degrees=30)],
    "translate_x": [v2.CenterCrop(224), v2.RandomAffine(degrees=0, translate=[0.2, 0])],
    "translate_y": [v2.CenterCrop(224), v2.RandomAffine(degrees=0, translate=[0, 0.2])],
    "shear_x": [v2.CenterCrop(224), v2.RandomAffine(degrees=0, shear=[0.0, 30.0])],
    "shear_y": [
        v2.CenterCrop(224),
        v2.RandomAffine(degrees=0, shear=[0.0, 0.0, 0.0, 30.0]),
    ],
    "elastic_transform": [v2.CenterCrop(224), v2.ElasticTransform()],
    "brightness": [v2.CenterCrop(224), v2.ColorJitter(brightness=0.5)],
    "contrast": [v2.CenterCrop(224), v2.ColorJitter(contrast=0.5)],
    "saturation": [v2.CenterCrop(224), v2.ColorJitter(saturation=0.5)],
    "gaussian_blur": [v2.CenterCrop(224), v2.GaussianBlur(kernel_size=3)],
    # "equalize": [v2.CenterCrop(224), v2.RandomEqualize()],
    # "gaussian_noise": [
    #     v2.CenterCrop(224),
    #     monai_transforms.RandGaussianNoise(prob=0.5),
    #     v2.GaussianNoise()
    # ],
    "scaling": [
        v2.CenterCrop(224),
        v2.RandomAffine(degrees=0, scale=[0.8, 1.2]),
    ],  # Sensible range based on prior works
}


def transform(image_size, aug_level, crop=False):
    if aug_level >= 0:
        augmentation = []
    if aug_level >= 1:
        candidates = ["rotation", "translate_x", "translate_y"]
        pick = random.randint(0, len(candidates)-1)
        augmentation += AUGMENTATIONS[candidates[pick]]
    if aug_level >= 2:
        candidates = ["shear_x", "shear_y", "elastic_transform", "horizontal_flip", "vertical_flip"]
        pick = random.randint(0, len(candidates)-1)
        augmentation += AUGMENTATIONS[candidates[pick]]
    if aug_level >= 3: 
        candidates = ["brightness", "contrast", "saturation", "gaussian_blur", "scaling"]
        pick = random.randint(0, len(candidates)-1)
        augmentation += AUGMENTATIONS[candidates[pick]]
    
    crop = [v2.CenterCrop(image_size)] if crop else [v2.Resize(image_size, antialias=True)]
        
    return v2.Compose(
        [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        + crop
        + augmentation
        + [v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

class ACROBAT(Dataset):
    '''Original ACROBAT dataset for Classification'''
    HER2_LEVELS = {
        'neg': 0,
        '1': 1,
        '2': 2,
        '3': 3, 
        'tissue': 4 # class added : tissue region
    }
    def __init__(self, type='train', image_size=224, aug_level=0, magnification='high'):
        self.HER2data = []
        self.labels = []
        self.numbers = [] # patient number
        self.type = type # train or test
        self.aug_level = aug_level if type == 'train' else 0
        self.image_size = image_size
        if magnification == 'high':
            file_paths = os.path.join(path.get_patch_dir(type='pyramid'), f'*/*_HER2_{type}/*/*.jpeg')
        else:
            file_paths = os.path.join(path.get_patch_dir(type='pyramid'), f'*/*_HER2_{type}/*.jpeg')
        print('raw data path:', file_paths)
        self.HER2_file_list = sorted(glob.glob(file_paths))
        assert len(self.HER2_file_list) > 0, 'HER2 file list is empty'

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        for HER2_path in tqdm(self.HER2_file_list, desc=f'Loading {type} dataset'):
            HER2_path_split = HER2_path.rstrip('.jpeg').split(os.sep) 
            if magnification == 'high':
                label, name = HER2_path_split[-4], HER2_path_split[-3] # {data_dir}/{single or pyramid}/{label}/{{number}_{img_type}_{type}}/{{row}_{col}}/{{row}_{col}}.jpeg
            else:
                label, name = HER2_path_split[-3], HER2_path_split[-2] # {data_dir}/{single or pyramid}/{label}/{{number}_{img_type}_{type}}/{{row}_{col}}.jpeg
            number, _, _ = name.split('_')
            image = Image.open(HER2_path) # 224 x 224
            self.HER2data.append(image)
            self.labels.append(self.HER2_LEVELS[label])
            self.numbers.append(number)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        t = transform(self.image_size, self.aug_level)
        HER2data = t(self.HER2data[idx])
        sample = {'IHC':HER2data, 'label': self.labels[idx]}
        return sample

    
def get_acrobat_dataloaders(type='classification', 
                            batch_size=32, 
                            num_workers=4, 
                            image_size=256,
                            aug_level=0)->dict:
    '''
    get dataloaders for BCI dataset.
    input :
        type : classification or segmentation
        batch_size : batch size
        num_workers : number of workers for dataloader
        image_size : image size
    output :
        dataloaders : dataloaders for train, validation, test
    '''
    if type == 'classification':
        train_dataset = ACROBAT(type='train', image_size=image_size, aug_level=aug_level, magnification='high')
        test_dataset = ACROBAT(type='val', image_size=image_size, magnification='high')
        train_shuffle = True
    else:
        raise ValueError(f'Invalid type: {type}. Only classification is supported.')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return {'train': train_loader, 'val': test_loader, 'test': test_loader}

