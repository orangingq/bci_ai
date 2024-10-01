import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm import tqdm
import albumentations as albu
from albumentations.pytorch import ToTensorV2

def transform(image_size):
    return albu.Compose([
        albu.Resize(image_size, image_size, always_apply=True),
        albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
        ToTensorV2(transpose_mask=True)
    ])

def crop_transform(image_size):
    return albu.Compose([
        albu.CenterCrop(image_size, image_size, always_apply=True),
        albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True),
        ToTensorV2(transpose_mask=True)
    ])

class HEDataset(Dataset):
    '''HE Image dataset for Segmentation'''
    HER2_LEVELS = {
        '0': 0,
        '1+': 1,
        '2+': 2,
        '3+': 3,
        'tissue': 4 # class added : tissue region
    }
    def __init__(self, data_dir='BCI_dataset', type='train', image_size=256):
        self.HEdata = [] # HE image
        self.numbers = [] # patient number
        self.type = type # train or test
        self.transform1 = transform(image_size)
        self.transform2 = crop_transform(image_size)
        self.directory = os.path.join(data_dir, 'HE', type)
        assert os.path.exists(self.directory), f'{self.directory} does not exist'
        self.filenames = sorted(os.listdir(self.directory)) # HE image filenames

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        for file_name in tqdm(self.filenames):
            number, _, _ = file_name.rstrip('.png').split('_') # (number)_(train/test)_(HER2level).png
            HE_file_path = os.path.join(self.directory, file_name)
            HE_image = Image.open(HE_file_path) # 1024 x 1024
            HE_image_rotated = HE_image.rotate(90)
            HE_image_rotated2 = HE_image.rotate(180)
            self.HEdata.append(HE_image)
            self.numbers.append(number)
            self.HEdata.append(HE_image_rotated)
            self.numbers.append(number)
            self.HEdata.append(HE_image_rotated2)
            self.numbers.append(number)
        return

    def __len__(self):
        return len(self.numbers)

    def __getitem__(self, idx):
        HEdata = self.transform1(image=np.asarray(self.HEdata[idx]))
        HEdata2 = self.transform2(image=np.asarray(self.HEdata[idx]))
        sample = {'HE': HEdata['image'], 'HE2': HEdata2['image'], 'number': self.numbers[idx], 'filename': self.filenames[idx]}
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
    def __init__(self, data_dir='BCI_dataset', type='train', image_size=224):
        self.HEdata = []
        self.IHCdata = []
        self.labels = []
        self.numbers = [] # patient number
        self.type = type # train or test
        self.transform = transform(image_size)
        HE_dir = os.path.join(data_dir, 'HE', type)
        IHC_dir = os.path.join(data_dir, 'IHC', type)
        assert os.path.exists(HE_dir) and os.path.exists(IHC_dir), f'{HE_dir} or {IHC_dir} does not exist'
        assert os.listdir(HE_dir) == os.listdir(IHC_dir)
        file_list = sorted(os.listdir(HE_dir))

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        for file_name in file_list:
            number, _, HER2label = file_name.rstrip('.png').split('_') # (number)_(train/test)_(HER2level).png
            HE_file_path, IHC_file_path = os.path.join(HE_dir, file_name), os.path.join(IHC_dir, file_name)
            HE_image, IHC_image = Image.open(HE_file_path), Image.open(IHC_file_path) # 1024 x 1024
            for theta in [0, 90, 180]: # data augmentation : rotation
                self.HEdata.append(HE_image.rotate(theta))
                self.IHCdata.append(IHC_image)
                self.labels.append(self.HER2_LEVELS[HER2label])
                self.numbers.append(number)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        HEdata = self.transform(image=np.asarray(self.HEdata[idx]))
        IHCdata = self.transform(image=np.asarray(self.IHCdata[idx]))
        sample = {'HE': HEdata['image'], 'IHC': IHCdata['image'], 'label': self.labels[idx]}
        return sample

def get_bci_dataloaders(data_dir='BCI_dataset', type='classification', batch_size=32, num_workers=4, image_size=256)->dict:
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
        train_dataset = BCIDataset(data_dir=data_dir, type='train', image_size=image_size)
        test_dataset = BCIDataset(data_dir=data_dir, type='test', image_size=image_size)
        train_shuffle = True
    elif type == 'segmentation':
        train_dataset = HEDataset(data_dir=data_dir, type='train', image_size=image_size)
        test_dataset = HEDataset(data_dir=data_dir, type='test', image_size=image_size)
        train_shuffle = False
    else:
        raise ValueError(f'Invalid type: {type}. Choose from [classification, segmentation]')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return {'train': train_loader, 'val': test_loader, 'test': test_loader}

