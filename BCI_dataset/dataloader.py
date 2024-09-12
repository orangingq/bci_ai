import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from torchvision import transforms


class BCIDataset(Dataset):
    HER2_LEVELS = {
        '0': 0,
        '1+': 1,
        '2+': 2,
        '3+': 3
    }
    def __init__(self, data_dir='BCI_dataset', type='train', transform=None):
        self.HEdata = []
        self.IHCdata = []
        self.labels = []
        self.type = type # train or test
        self.transform = transform
        HE_dir = os.path.join(data_dir, 'HE', type)
        IHC_dir = os.path.join(data_dir, 'IHC', type)
        assert os.path.exists(HE_dir) and os.path.exists(IHC_dir), f'{HE_dir} or {IHC_dir} does not exist'
        assert os.listdir(HE_dir) == os.listdir(IHC_dir)
        file_list = os.listdir(HE_dir)

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        for file_name in file_list:
            number, _, HER2label = file_name.rstrip('.png').split('_') # (number)_(train/test)_(HER2level).png
            HE_file_path, IHC_file_path = os.path.join(HE_dir, file_name), os.path.join(IHC_dir, file_name)
            HE_image, IHC_image = Image.open(HE_file_path), Image.open(IHC_file_path) # 1024 x 1024
            self.HEdata.append(HE_image)
            self.IHCdata.append(IHC_image)
            self.labels.append(self.HER2_LEVELS[HER2label])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.transform:
            HEdata = self.transform(self.HEdata[idx])
            IHCdata = self.transform(self.IHCdata[idx])
        sample = {'HE': HEdata, 'IHC': IHCdata, 'label': self.labels[idx]}
        return sample

def get_bci_dataloaders(data_dir='BCI_dataset', batch_size=32, num_workers=4, image_size=224):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = BCIDataset(data_dir=data_dir, type='train', transform=transform)
    test_dataset = BCIDataset(data_dir=data_dir, type='test', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return {'train': train_loader, 'val': test_loader, 'test': test_loader}

