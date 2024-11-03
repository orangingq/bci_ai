import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import ViTFeatureExtractor, ViTForImageClassification
from torchvision import models
from sklearn.metrics import roc_curve, auc

from datasets import CustomDataset, augmentations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random_seed = 3565#random.randint(0, 10000)
print(f"Generated random seed: {random_seed}")#4750#3565
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

# 테스트 모델 함수
def test_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    total = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 모델 예측
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1) #ViT: outputs.logits
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    return np.array(all_labels), np.array(all_predictions), correct, total


# 데이터 경로 설정
data_folders = [ os.path.join('/pathology/bci/', 'BCI_dataset', 'IHC', 'train'), os.path.join('/pathology/bci/', 'BCI_dataset', 'IHC', 'test') ]
csv_files = [ os.path.join('/pathology/bci/', 'BCI_train_label_1027.csv'), os.path.join('/pathology/bci/', 'BCI_test_label_1027.csv') ]
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

# 데이터셋 생성
dataset = CustomDataset(data_folders, csv_files, feature_extractor)

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

'''
#ViT 결과 확인
model_name = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor(model_name)

#model_path = '/pathology/bci/model/vision_model_1030_newsplit_test884.pth'
model_path = '/pathology/bci/model/vision_model_1101_3_1e-05_acc86.pth'

model = ViTForImageClassification.from_pretrained(model_name, num_labels=5).to(device)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# 모델 테스트 및 평가
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
results_df.to_csv('ViT_pred.csv', index=False)

# Confusion Matrix 생성 및 저장
unique_labels = np.unique(true_labels)
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=unique_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")

# Confusion Matrix 파일 저장
plt.savefig('ViT_confusion_matrix.png')
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
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')

# ROC 곡선 파일 저장
plt.savefig('ViT_roc_curve.png')
plt.show()

'''

# 모델 경로 및 설정
model_name = 'resnet50'
model_path = '/pathology/bci/model/ResNet50_1102_test8383.pth'

# ResNet50 모델 로드 및 가중치 설정
model = models.resnet50(pretrained=True)  # pretrained 모델 로드
num_classes = 5
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # fc 레이어 수정

# pretrained weights를 사용하여 로드 (strict=False로)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)


# 모델 테스트 및 평가
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