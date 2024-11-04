#todo: 코드 정리 (중복되는 함수 하나로 묶기)

import os
import argparse
import random
import torch
import cv2

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2
import torchvision.models as models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc

from utils import EarlyStopping
from datasets import ResNetDataset, augmentations


parser = argparse.ArgumentParser(description="Script for training a model with data augmentation.")
parser.add_argument('--root_dir', type=str, default='/pathology/bci/',
                    help='Directory to save the model')
parser.add_argument('--augmentation_level', type=int, choices=[0, 1, 2, 3], default=3,
                    help='Select the augmentation level (0, 1, 2, 3)')
parser.add_argument('--lr', type=float, default=1e-5,
                    help='learning rate')

args = parser.parse_args()

num_classes=5

# GPU 사용을 위한 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 경로 설정
data_folders = [
    os.path.join(args.root_dir, 'BCI_dataset', 'IHC', 'train'),
    os.path.join(args.root_dir, 'BCI_dataset', 'IHC', 'test')
]
csv_files = [
    os.path.join(args.root_dir, 'BCI_train_label_1027.csv'),
    os.path.join(args.root_dir, 'BCI_test_label_1027.csv')
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

# 데이터셋 생성
dataset = ResNetDataset(data_folders, csv_files, transform, augmentations=augmentations[args.augmentation_level])

# train, val, test 데이터셋 분할
indices = list(range(len(dataset)))
labels = [dataset.labels[idx] for idx in indices]

train_indices, temp_indices, _, temp_labels = train_test_split(
    indices, labels, test_size=0.3, stratify=labels, random_state=42)

val_indices, test_indices, _, _ = train_test_split(
    temp_indices, temp_labels, test_size=0.33, stratify=temp_labels, random_state=42)

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




# ResNet 모델 및 옵티마이저 설정
model = models.resnet50(pretrained=True)  
num_classes = 5  
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  
model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)

criterion = nn.CrossEntropyLoss()

num_epochs = 20
patience = 5

model_save_path = os.path.join(args.root_dir, 'model', 'resnet50_1102.pth')
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

        outputs = model(images)
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

            outputs = model(images)
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


model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Valid Accuracy: {accuracy:.2f}%")


total = 0
correct = 0

def test_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    global total, correct

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 모델 예측
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    return np.array(all_labels), np.array(all_predictions)

# 평가 및 정확도 계산
true_labels, predicted_labels = test_model(model, test_loader, device)

accuracy = 100 * correct / total
print(f"Valid Accuracy: {accuracy:.2f}%")

# 정답 라벨별 정확도 출력
unique_labels = np.unique(true_labels)
for label in unique_labels:
    mask = true_labels == label
    accuracy = accuracy_score(true_labels[mask], predicted_labels[mask])
    print(f"Accuracy for label {label}: {accuracy * 100:.2f}%")

# Confusion Matrix 출력 및 시각화
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=unique_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

##
true_labels, predicted_labels, correct, total = test_model(model, test_loader, device)

# 성능 지표 계산
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')

# 출력
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# 예측 결과 CSV 저장
results_df = pd.DataFrame({'Number': np.arange(len(predicted_labels)), 'True_Label': true_labels, 'Predicted_Label': predicted_labels})
results_df.to_csv('ResNet50_pred.csv', index=False)

# Confusion Matrix 생성 및 저장
unique_labels = np.unique(true_labels)
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=unique_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")

# Confusion Matrix 파일 저장
plt.savefig('ResNet50_confusion_matrix.png')
plt.show()

# ROC 곡선 그리기
# 예측 확률 계산
model.eval()  # 모델을 평가 모드로 전환
predicted_probs = []

with torch.no_grad():  # 그래디언트 계산 비활성화
    for inputs, _ in test_loader:  # 테스트 데이터로부터 입력을 가져옴
        inputs = inputs.to(device)  # 입력을 GPU로 이동
        outputs = model(inputs)  # 모델에 입력
        probs = torch.softmax(outputs.logits, dim=1)  # 소프트맥스를 적용하여 확률 계산
        predicted_probs.append(probs.cpu().numpy())  # CPU로 이동 후 리스트에 추가

predicted_probs = np.vstack(predicted_probs)  # 리스트를 NumPy 배열로 변환

# ROC 곡선 및 AUC 계산
n_classes = len(unique_labels)
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(true_labels, predicted_probs[:, i], pos_label=i)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label='Class {} (AUC = {:.2f})'.format(i, roc_auc))

plt.plot([0, 1], [0, 1], 'k--', lw=2)  # 대각선
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ResNet50 ROC Curve')
plt.legend(loc='lower right')

# ROC 곡선 파일 저장
plt.savefig('ResNet50_roc_curve.png')
plt.show()