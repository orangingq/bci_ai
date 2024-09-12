import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

def get_file_paths(folder):
    image_file_paths = []
    for root, dirs, filenames in os.walk(folder):
        filenames = sorted(filenames)
        for filename in filenames:
            input_path = os.path.abspath(root)
            file_path = os.path.join(input_path, filename)
            if filename.endswith('.png') or filename.endswith('.jpg'):
                image_file_paths.append(file_path)

        break  # prevent descending into subfolders
    return image_file_paths


def align_images(a_file_paths, b_file_paths, target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for i in range(len(a_file_paths)):
        img_a = Image.open(a_file_paths[i])
        img_b = Image.open(b_file_paths[i])
        assert(img_a.size == img_b.size)

        aligned_image = Image.new("RGB", (img_a.size[0] * 2, img_a.size[1]))
        aligned_image.paste(img_a, (0, 0))
        aligned_image.paste(img_b, (img_a.size[0], 0))
        aligned_image.save(os.path.join(target_path, '{:04d}.jpg'.format(i)))


class BCIDataset(Dataset):
    def __init__(self, data_dir=''):
        self.data = []
        self.labels = []
        for file_name in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file_name)
            data = np.load(file_path)
            self.data.append(data['features'])
            self.labels.append(data['labels'])
        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        'number_train/test_HER2level.png'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {'feature': self.data[idx], 'label': self.labels[idx]}
        return sample

def get_bci_dataloaders(data_dir, batch_size=32, num_workers=4):
    train_dataset = BCIDataset(os.path.join(data_dir, 'train'))
    test_dataset = BCIDataset(os.path.join(data_dir, 'test'))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return {'train': train_loader, 'test': test_loader}
