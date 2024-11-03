import os
import argparse
import random
import torch
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2
import pandas as pd

from collections import Counter
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification


#Augmentations
AUGMENTATIONS = {
    "horizontal_flip": [v2.RandomHorizontalFlip()],
    "vertical_flip": [v2.RandomVerticalFlip()],
    "rotation": [v2.RandomRotation(degrees=30)],
    "translate_x": [v2.RandomAffine(degrees=0, translate=[0.2, 0])],
    "translate_y": [v2.RandomAffine(degrees=0, translate=[0, 0.2])],
    "shear_x": [v2.RandomAffine(degrees=0, shear=[0.0, 30.0])],
    "shear_y": [
        v2.RandomAffine(degrees=0, shear=[0.0, 0.0, 0.0, 30.0]),
    ],
    "elastic_transform": [v2.ElasticTransform()],
    "brightness": [v2.ColorJitter(brightness=0.5)],
    "contrast": [v2.ColorJitter(contrast=0.5)],
    "saturation": [v2.ColorJitter(saturation=0.5)],
    "gaussian_blur": [v2.GaussianBlur(kernel_size=3)],
    "scaling": [
        v2.RandomAffine(degrees=0, scale=[0.8, 1.2]),
    ], 
}
augmentations = {
    1: v2.Compose([random.choice(AUGMENTATIONS["rotation"] + AUGMENTATIONS["translate_x"] + AUGMENTATIONS["translate_y"] + AUGMENTATIONS["horizontal_flip"] + AUGMENTATIONS["vertical_flip"])]),
    2: v2.Compose([random.choice(AUGMENTATIONS["rotation"] + AUGMENTATIONS["translate_x"] + AUGMENTATIONS["translate_y"] + AUGMENTATIONS["shear_x"] + AUGMENTATIONS["shear_y"] + AUGMENTATIONS["elastic_transform"] + AUGMENTATIONS["horizontal_flip"] + AUGMENTATIONS["vertical_flip"])]),
    3: v2.Compose([random.choice(AUGMENTATIONS["rotation"] + AUGMENTATIONS["translate_x"] + AUGMENTATIONS["translate_y"] + AUGMENTATIONS["shear_x"] + AUGMENTATIONS["shear_y"] + AUGMENTATIONS["elastic_transform"] + AUGMENTATIONS["horizontal_flip"] + AUGMENTATIONS["vertical_flip"] + AUGMENTATIONS["brightness"] + AUGMENTATIONS["contrast"] + AUGMENTATIONS["saturation"] + AUGMENTATIONS["gaussian_blur"] + AUGMENTATIONS["scaling"])])
}


#ViT
class CustomDataset(Dataset):
    def __init__(self, folder, csv, feature_extractor, augmentations=None):
        self.folder1 = folder[0]
        self.folder2 = folder[1]
        self.feature_extractor = feature_extractor
        self.augmentations = augmentations

        # CSV 파일 읽기
        self.csv1 = pd.read_csv(csv[0], encoding='ISO-8859-1')
        self.csv2 = pd.read_csv(csv[1], encoding='ISO-8859-1')

        # 'Number' 열의 값을 문자열 형식으로 다섯 자리로 변환하여 사용
        self.csv1['Number'] = self.csv1['Number'].apply(lambda x: str(x).zfill(5))
        self.csv2['Number'] = self.csv2['Number'].apply(lambda x: str(x).zfill(5))

        # 이미지 파일과 라벨 매핑을 위한 딕셔너리 생성
        self.label_dict1 = dict(zip(self.csv1['Number'], self.csv1['1025_Re_labeled']))
        self.label_dict2 = dict(zip(self.csv2['Number'], self.csv2['1025_Re_labeled']))

        # 이미지 파일 경로 리스트 생성
        self.image_paths = []
        self.labels = []

        # folder1 데이터 가져오기 (BCI train)
        for image_file in os.listdir(self.folder1):
            if image_file.endswith('.png'):
                image_number = image_file.split('_')[0]

                if image_number in self.label_dict1:
                    self.image_paths.append(os.path.join(self.folder1, image_file))
                    self.labels.append(self.label_dict1[image_number])
                else:
                    print(f"Warning: Label for {image_file} not found in CSV and is skipped.")
        
        # folder2 데이터 가져오기 (BCI test)
        for image_file in os.listdir(self.folder2):
            if image_file.endswith('.png'):
                image_number = image_file.split('_')[0]

                if image_number in self.label_dict2:
                    self.image_paths.append(os.path.join(self.folder2, image_file))
                    self.labels.append(self.label_dict2[image_number])
                else:
                    print(f"Warning: Label for {image_file} not found in CSV and is skipped.")

        # folder3 데이터 가져오기 (acrobat)
        for label in ["0", "1", "0_add", "3_add"]:
            label_folder = os.path.join('/pathology/acrobat_patch', str(label))
            for image_file in os.listdir(label_folder):
                if image_file.endswith('.jpeg'):
                    self.image_paths.append(os.path.join(label_folder, image_file))
                    self.labels.append(int(label.split('_')[0]))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")

        if self.augmentations:
            image = self.augmentations(image)
                
        inputs = self.feature_extractor(images=image, return_tensors="pt")  # 이미지 전처리

        return inputs["pixel_values"].squeeze(0), label


#ResNet
class ResNetDataset(Dataset):
    def __init__(self, folder, csv, transform, augmentations=None):
        self.folder1 = folder[0]
        self.folder2 = folder[1]
        self.transform = transform
        self.augmentations = augmentations

        # CSV 파일 읽기
        self.csv1 = pd.read_csv(csv[0], encoding='ISO-8859-1')
        self.csv2 = pd.read_csv(csv[1], encoding='ISO-8859-1')

        # 'Number' 열의 값을 문자열 형식으로 다섯 자리로 변환하여 사용
        self.csv1['Number'] = self.csv1['Number'].apply(lambda x: str(x).zfill(5))
        self.csv2['Number'] = self.csv2['Number'].apply(lambda x: str(x).zfill(5))

        # 이미지 파일과 라벨 매핑을 위한 딕셔너리 생성
        self.label_dict1 = dict(zip(self.csv1['Number'], self.csv1['1025_Re_labeled']))
        self.label_dict2 = dict(zip(self.csv2['Number'], self.csv2['1025_Re_labeled']))

        # 이미지 파일 경로 리스트 생성
        self.image_paths = []
        self.labels = []

        # folder1 데이터 가져오기 (BCI train)
        for image_file in os.listdir(self.folder1):
            if image_file.endswith('.png'):
                image_number = image_file.split('_')[0]

                if image_number in self.label_dict1:
                    self.image_paths.append(os.path.join(self.folder1, image_file))
                    self.labels.append(self.label_dict1[image_number])
                else:
                    print(f"Warning: Label for {image_file} not found in CSV and is skipped.")
        
        # folder2 데이터 가져오기 (BCI test)
        for image_file in os.listdir(self.folder2):
            if image_file.endswith('.png'):
                image_number = image_file.split('_')[0]

                if image_number in self.label_dict2:
                    self.image_paths.append(os.path.join(self.folder2, image_file))
                    self.labels.append(self.label_dict2[image_number])
                else:
                    print(f"Warning: Label for {image_file} not found in CSV and is skipped.")

        # folder3 데이터 가져오기 (acrobat)
        for label in ["0", "1", "0_add", "3_add"]:
            label_folder = os.path.join('/pathology/acrobat_patch', str(label))
            for image_file in os.listdir(label_folder):
                if image_file.endswith('.jpeg'):
                    self.image_paths.append(os.path.join(label_folder, image_file))
                    self.labels.append(int(label.split('_')[0]))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")

        if self.augmentations:
            image = self.augmentations(image)
                
        if self.transform:
            image = self.transform(image)

        return image, label


# 데이터셋 별 라벨 수 출력
# train_label_counts = Counter([label for _, label in train_dataset])
# val_label_counts = Counter([label for _, label in val_dataset])
# test_label_counts = Counter([label for _, label in test_dataset])

# for label, count in sorted(train_label_counts.items()):
#     print(f"Train Label {label}: {count}")
# for label, count in sorted(val_label_counts.items()):
#     print(f"Val Label {label}: {count}")
# for label, count in sorted(test_label_counts.items()):
#     print(f"Test Label {label}: {count}")
