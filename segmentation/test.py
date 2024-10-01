import sys
import os

import cv2
import torch
from albumentations.pytorch import ToTensorV2
import albumentations as albu
from src.models.hooknet import HookNet
from tools.evaluate import BCSS_CLASSES

#0. src 폴더 경로를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

#1. 이미지 불러오기
def load_image(image_path, transforms):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 이미지 변환(transform)을 적용
    augmented = transforms(image=image)
    return augmented['image'].unsqueeze(0)

#2. segmentation mask 생성
def predict_segmentation(model, image_tensor1, image_tensor2):
    model.eval()
    with torch.no_grad():
        preds_context, preds_target = model(image_tensor1, image_tensor2)
        pred_mask = torch.argmax(preds_target, dim=1)  # 클래스별로 가장 높은 확률을 가진 픽셀
    print("Context mask shape:", preds_context.shape)
    print("Target mask shape:", preds_target.shape)
    return pred_mask.cpu().numpy()

#3. state_dict 키 이름 수정하는 함수 정의
from collections import OrderedDict

def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[len('module.'):]  # 'module.' prefix 제거
        new_state_dict[k] = v
    return new_state_dict

#4. segmentation visualization
import matplotlib.pyplot as plt
import numpy as np

def visualize_segmentation(seg_mask, num_classes):
    colors = plt.get_cmap('tab20', num_classes) 
    seg_mask_color = colors(seg_mask[0]) 

    plt.imshow(seg_mask_color)
    plt.colorbar()
    plt.show()

    seg_mask_color = (seg_mask_color * 255).astype(np.uint8)
    
    return seg_mask_color

def save_segmentation(seg_mask_color, save_path):
    cv2.imwrite(save_path, cv2.cvtColor(seg_mask_color, cv2.COLOR_RGBA2BGRA)) 

#5. inference 수행
def main_inference(image_path, model_path, save_path):
    #1) 모델 불러오기
    model = HookNet(encoder_name="resnet18", encoder_weights="imagenet", classes=len(BCSS_CLASSES) + 1)
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = remove_module_prefix(checkpoint["state_dict"])
    model.load_state_dict(state_dict)

    #2) 이미지 변환 정의 (resize, centercrop)
    transform1 = albu.Compose([
        albu.Resize(256, 256, always_apply=True),
        albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(transpose_mask=True)
    ])
    transform2 = albu.Compose([
        albu.CenterCrop(256, 256, always_apply=True),
        albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(transpose_mask=True)
    ])

    #3) 두 가지 해상도의 이미지 불러오기 및 transform 적용
    image_tensor1 = load_image(image_path, transform1)  # 고해상도 (resize 적용)
    image_tensor2 = load_image(image_path, transform2)  # 저해상도 (centercrop 적용)

    #4) Segmentation Mask 생성
    seg_mask = predict_segmentation(model, image_tensor1, image_tensor2)

    print("Segmentation mask min value:", seg_mask.min())
    print("Segmentation mask max value:", seg_mask.max())
    
    #5) 디버깅을 위한 시각화
    seg_mask_color=visualize_segmentation(seg_mask, num_classes=5)
    save_segmentation(seg_mask_color, save_path)
    print(f"Segmentation 결과가 저장되었습니다.")

image_path='C:/Users/82103/msf-wsi/00000_test_1.png'
model_path='C:/Users/82103/msf-wsi/bcss_fold0_ft_model.pth.tar'
save_path='C:/Users/82103/msf-wsi/results/segmentation.png'

main_inference(image_path, model_path, save_path)


# # 현재 모델의 키 확인
# model_keys = model.state_dict().keys()
# print("Model keys:", model_keys)

# # 저장된 가중치의 키 확인
# checkpoint = torch.load('C:/Users/82103/msf-wsi/bcss_fold0_ft_model.pth.tar', map_location=torch.device('cpu'))["state_dict"]
# checkpoint_keys = checkpoint.keys()
# print("Checkpoint keys:", checkpoint_keys)
