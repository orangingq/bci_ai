import os
import argparse
import random
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as v2
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
from transformers import ViTFeatureExtractor, ViTForImageClassification
from sklearn.metrics import confusion_matrix, accuracy_score

from datasets import CustomDataset, augmentations
from utils import EarlyStopping

random_seed = 3565#random.randint(0, 10000)

random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

parser = argparse.ArgumentParser(description="Script for training a model with data augmentation.")
parser.add_argument('--root_dir', type=str, default='/pathology/',
                    help='Directory to save the model')
parser.add_argument('--augmentation_level', type=int, choices=[0, 1, 2, 3], default=3,
                    help='Select the augmentation level (0, 1, 2, 3)')
parser.add_argument('--lr', type=float, default=1e-5,
                    help='learning rate')
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 데이터 경로 설정
data_folders = [ os.path.join(args.root_dir, 'bci', 'train'), os.path.join(args.root_dir, 'bci', 'test'), os.path.join(args.root_dir, 'acrobat_patch') ]
csv_files = [ os.path.join(args.root_dir, 'BCI_train_label.csv'), os.path.join(args.root_dir, 'BCI_test_label.csv') ]
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

# 데이터셋 생성
dataset = CustomDataset(data_folders, csv_files, feature_extractor, augmentations=augmentations[args.augmentation_level])

# train, val, test 데이터셋 분할
indices = list(range(len(dataset)))
labels = [dataset.labels[idx] for idx in indices]

train_indices, temp_indices, _, temp_labels = train_test_split(
    indices, labels, test_size=0.3, stratify=labels, random_state=42)

val_indices, test_indices, _, _ = train_test_split(
    temp_indices, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)

train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

# 데이터 로더 생성
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train set size: {len(train_dataset)} images")
print(f"Validation set size: {len(val_dataset)} images")
print(f"Test set size: {len(test_dataset)} images")

# ViT 모델 및 옵티마이저 설정
model_name = 'google/vit-base-patch16-224-in21k'
model = ViTForImageClassification.from_pretrained(model_name, num_labels=5).to(device)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)

criterion = nn.CrossEntropyLoss()

num_epochs = 20
patience = 5

model_save_path = os.path.join(args.root_dir, 'model', f'vision_model_{args.augmentation_level}_{args.lr}.pth')
early_stopping = EarlyStopping(patience=patience, path=model_save_path)

# 모델 학습
for epoch in range(num_epochs):
    # Train loss, accuracy
    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Train accuracy 계산
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

    train_loss /= len(train_loader)
    train_accuracy = train_correct / train_total

    # Validation loss
    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():  # 그래디언트 계산 중지
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images).logits
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            # Validation accuracy 계산
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(val_loader)
    val_accuracy = val_correct / val_total

    # 에폭마다 학습 및 검증 손실과 정확도 출력
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
          f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_accuracy:.4f}")

    # Early Stopping 체크
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

# 최적의 모델 불러오기
early_stopping.load_best_model(model)
print("Best model restored.")
