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
            _, predicted = torch.max(outputs.logits, 1) #ViT: outputs.logits
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    return np.array(all_labels), np.array(all_predictions), correct, total


# 데이터 경로 설정
data_folders = [ os.path.join('/pathology/', 'bci', 'train'), os.path.join('/pathology/', 'bci', 'test') ]
csv_files = [ os.path.join('/pathology/', 'BCI_train_label.csv'), os.path.join('/pathology/', 'BCI_test_label.csv') ]
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

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize

# 모델 정의 및 로드
model_name = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor(model_name)
model_path = '/pathology/bci/model/vision_model_1103_3_1e-05.pth'

model = ViTForImageClassification.from_pretrained(model_name, num_labels=5).to(device)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# 모델 테스트 및 평가
true_labels, predicted_labels, correct, total = test_model(model, test_loader, device)

# 예측 확률 계산
model.eval()
predicted_probs = []

with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        predicted_probs.append(probs.cpu().numpy())

predicted_probs = np.vstack(predicted_probs)

# 성능 지표 계산
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')
overall_auc = roc_auc_score(label_binarize(true_labels, classes=np.arange(5)), predicted_probs, average='weighted', multi_class='ovr')

print(f"Overall Accuracy: {accuracy * 100:.2f}%")
print(f"Overall Precision: {precision:.4f}")
print(f"Overall Recall: {recall:.4f}")
print(f"Overall F1-Score: {f1:.4f}")
print(f"Overall AUC: {overall_auc:.4f}")

# 각 클래스별 성능 지표 계산
unique_labels = np.unique(true_labels)
true_labels_one_hot = label_binarize(true_labels, classes=np.arange(5))

for i in unique_labels:
    class_precision = precision_score(true_labels, predicted_labels, labels=[i], average='weighted', zero_division=0)
    class_recall = recall_score(true_labels, predicted_labels, labels=[i], average='weighted', zero_division=0)
    class_f1 = f1_score(true_labels, predicted_labels, labels=[i], average='weighted', zero_division=0)
    class_mask = (true_labels == i)
    class_accuracy = np.sum((predicted_labels[class_mask] == true_labels[class_mask])) / np.size(class_mask)
    class_auc = roc_auc_score(true_labels_one_hot[:, i], predicted_probs[:, i])

    print(f"\nClass {i} Metrics:")
    print(f"Accuracy: {class_accuracy:.4f}")
    print(f"Precision: {class_precision:.4f}")
    print(f"Recall: {class_recall:.4f}")
    print(f"F1-Score: {class_f1:.4f}")
    print(f"AUC: {class_auc:.4f}")

# Confusion Matrix 생성 및 시각화
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=unique_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig('ViT_confusion_matrix.png')
plt.show()

# ROC 곡선 그리기
plt.figure(figsize=(12, 8))
for i in range(len(unique_labels)):
    fpr, tpr, _ = roc_curve(true_labels_one_hot[:, i], predicted_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('ViT_roc_curve.png')
plt.show()
