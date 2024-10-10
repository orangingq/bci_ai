import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
DataLoader
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

class AcrobatDataset(Dataset):
    '''Original ACROBAT dataset for WSI Classification'''
    def __init__(self, data_dir='acrobat', type='train', image_size=224):
        self.HEdata = []
        self.labels = []
        self.numbers = [] # patient number
        self.type = type # train or test
        self.transform = transform(image_size)
        directory = os.path.join(data_dir, type)
        assert os.path.exists(directory), f'{directory} does not exist'
        file_length = max([name.split('_')[0] for name in os.listdir(directory)])+1

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        for number in tqdm(range(file_length), desc=f'Loading {type} dataset'):
            ER_path = os.path.join(directory, f'{number}_ER_{type}.tif')
            HE_path = os.path.join(directory, f'{number}_HE_{type}.tif')
            HER2_path = os.path.join(directory, f'{number}_HER2_{type}.tif')
            KI67_path = os.path.join(directory, f'{number}_KI67_{type}.tif')
            PGR_path = os.path.join(directory, f'{number}_PGR_{type}.tif')
            ER_img = Image.open(ER_path, formats=['TIF'])
            # HE_file_path, IHC_file_path = os.path.join(HE_dir, file_name), os.path.join(IHC_dir, file_name)
            # HE_image, IHC_image = Image.open(HE_file_path), Image.open(IHC_file_path) # 1024 x 1024
            if self.type == 'train':
                rotation = [0, 90, 180]
            else:
                rotation = [0]
            for theta in rotation: # data augmentation : rotation
                self.HEdata.append(HE_image.rotate(theta))
                self.IHCdata.append(IHC_image)
                assert self.label_dict[int(number)]['old'] == self.HER2_LEVELS[HER2label], f'Label mismatch: {number} {HER2label}'
                self.labels.append(self.label_dict[int(number)]['new'])
                self.numbers.append(number)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        HEdata = self.transform(image=np.asarray(self.HEdata[idx]))
        IHCdata = self.transform(image=np.asarray(self.IHCdata[idx]))
        sample = {'HE': HEdata['image'], 'IHC': IHCdata['image'], 'label': self.labels[idx]}
        return sample
    
    def new_label_dict(self)-> dict:
        filename = f'BCI_{self.type}_label.csv'
        label_dict = {}
        df = pd.read_csv(filename)
        for _, row in df.iterrows():
            number = int(row['Number'])
            label_dict[number] = {}
            label_dict[number]['old'] = int(row['Label'])
            label_dict[number]['new'] = int(row['Re_labeled'])
        assert len(self.file_list) == len(label_dict), 'Number of labels does not match number of images'
        return label_dict
    


def get_acrobat_dataloaders(data_dir='acrobat', type='classification', batch_size=32, num_workers=4, image_size=256)->dict:
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

