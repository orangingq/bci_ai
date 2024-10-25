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

data_dir = path.get_data_dir('BCI_dataset')

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


class HEDataset(Dataset):
    '''HE Image dataset for Segmentation'''
    HER2_LEVELS = {
        '0': 0,
        '1+': 1,
        '2+': 2,
        '3+': 3,
        'tissue': 4 # class added : tissue region
    }
    def __init__(self, type='train', image_size=256, aug_level=0):
        self.HEdata = [] # HE image
        self.numbers = [] # patient number
        self.type = type # train or test
        self.image_size = image_size
        self.aug_level = aug_level if type == 'train' else 0
        self.directory = os.path.join(data_dir, 'HE', type)
        assert os.path.exists(self.directory), f'{self.directory} does not exist'
        self.filenames = sorted(os.listdir(self.directory)) # HE image filenames

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        for file_name in tqdm(self.filenames):
            number, _, _ = file_name.rstrip('.png').split('_') # (number)_(train/test)_(HER2level).png
            HE_file_path = os.path.join(self.directory, file_name)
            HE_image = Image.open(HE_file_path) # 1024 x 1024
            self.HEdata.append(HE_image)
            self.numbers.append(number)
        return

    def __len__(self):
        return len(self.numbers)

    def __getitem__(self, idx):
        t1 = transform(self.image_size, self.aug_level)
        t2 = transform(self.image_size, self.aug_level, crop=True)
        HEdata = t1(self.HEdata[idx])
        HEdata2 = t2(self.HEdata[idx])
        sample = {'HE': HEdata, 'HE2': HEdata2, 'number': self.numbers[idx], 'filename': self.filenames[idx]}
        return sample


class BCIDataset(Dataset):
    '''Original BCI dataset for Classification'''
    HER2_LEVELS = {
        '0': 0,
        '1+': 1,
        '2+': 2,
        '3+': 3, 
        'tissue': 4 # class added : tissue region
    }
    def __init__(self, type='train', image_size=224, aug_level=0):
        self.HEdata = []
        self.IHCdata = []
        self.labels = []
        self.numbers = [] # patient number
        self.type = type # train or test
        self.aug_level = aug_level if type == 'train' else 0
        self.image_size = image_size
        HE_dir = os.path.join(data_dir, 'HE', type)
        IHC_dir = os.path.join(data_dir, 'IHC', type)
        assert os.path.exists(HE_dir) and os.path.exists(IHC_dir), f'{HE_dir} or {IHC_dir} does not exist'
        assert os.listdir(HE_dir) == os.listdir(IHC_dir)
        self.file_list = sorted(os.listdir(HE_dir))
        self.label_dict = self.new_label_dict()

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        for file_name in tqdm(self.file_list, desc=f'Loading {type} dataset'):
            number, _, HER2label = file_name.rstrip('.png').split('_') # (number)_(train/test)_(HER2level).png
            HE_file_path, IHC_file_path = os.path.join(HE_dir, file_name), os.path.join(IHC_dir, file_name)
            HE_image, IHC_image = Image.open(HE_file_path), Image.open(IHC_file_path) # 1024 x 1024
            self.HEdata.append(HE_image)
            self.IHCdata.append(IHC_image)
            assert self.label_dict[int(number)]['old'] == self.HER2_LEVELS[HER2label], f'Label mismatch: {number} {HER2label}'
            self.labels.append(self.label_dict[int(number)]['new'])
            self.numbers.append(number)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        t = transform(self.image_size, self.aug_level)
        HEdata = t(self.HEdata[idx])
        IHCdata = t(self.IHCdata[idx])
        sample = {'HE': HEdata, 'IHC': IHCdata, 'label': self.labels[idx]}
        return sample
    
    def new_label_dict(self)-> dict:
        filename = os.path.join(data_dir, f'BCI_{self.type}_label.csv')
        label_dict = {}
        df = pd.read_csv(filename)
        for _, row in df.iterrows():
            number = int(row['Number'])
            label_dict[number] = {}
            label_dict[number]['old'] = int(row['Label'])
            label_dict[number]['new'] = int(row['Re_labeled'])
        assert len(self.file_list) == len(label_dict), 'Number of labels does not match number of images'
        return label_dict
    

def get_bci_dataloaders(type='classification', 
                        batch_size=32, 
                        num_workers=4, 
                        image_size=256,
                        aug_level=0)->dict:
    '''
    get dataloaders for BCI dataset.
    input :
        data_dir : dataset directory
        type : classification or segmentation
        batch_size : batch size
        num_workers : number of workers for dataloader
        image_size : image size
    output :
        dataloaders : dataloaders for train, validation, test
    '''
    if type == 'classification':
        train_dataset = BCIDataset(type='train', image_size=image_size, aug_level=aug_level)
        test_dataset = BCIDataset(type='test', image_size=image_size)
        train_shuffle = True
    elif type == 'segmentation':
        train_dataset = HEDataset(type='train', image_size=image_size)
        test_dataset = HEDataset(type='test', image_size=image_size)
        train_shuffle = False
    else:
        raise ValueError(f'Invalid type: {type}. Choose from [classification, segmentation]')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return {'train': train_loader, 'val': test_loader, 'test': test_loader}

