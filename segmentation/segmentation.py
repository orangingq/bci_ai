
import os
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import cv2
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from segmentation.src.models.hooknet import HookNet
from segmentation.tools.evaluate import BCSS_CLASSES
from utils.util import random_seed, remove_module_prefix
from BCI_dataset.dataloader import get_bci_dataloaders

def predict_segmentation(dataloader, model):
    '''
    segmentation mask 생성
    input : 
        dataloader : dataloader
        model : segmentation model
    output :
        pred_mask : segmentation mask (True: tumor, False: background)
    '''
    model.eval()
    result = []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            image1, image2 = batch['HE'].cuda(), batch['HE2'].cuda()
            preds_context, preds_target = model(image1, image2)
            pred_mask = torch.argmax(preds_target, dim=1)  # 클래스별로 가장 높은 확률을 가진 픽셀
            pred_mask = pred_mask == 1 # 1번 클래스('tumor')가 가장 높은 확률을 가진 픽셀만 True (binary mask)
            result.append(pred_mask)
    result = torch.cat(result, dim=0).cpu().numpy()
    return result

# def visualize_segmentation(seg_mask, num_classes, filename, save_path):
#     cmap = mcolors.ListedColormap([(0, 1, 0, .21), 'green', 'yellow', 'orange', 'red'])
#     bounds = [i for i in range(1, num_classes+1)]  # Notice we go up to 6 to include the upper limit of 5
#     norm = mcolors.BoundaryNorm(bounds, cmap.N)
#     plt.imshow(seg_mask, cmap=cmap, norm=norm)
#     plt.colorbar(ticks=[i for i in range(num_classes)])
#     plt.show()
#     plt.savefig(save_path)
#     plt.close()
#     return 

def visualize_segmentation(seg_mask, filename, save_path):
    # Load the PNG image
    HE_image = Image.open(filename).convert("RGBA")  # Load the PNG image with transparency (if any)
    HE_img_array = np.array(HE_image)
    seg_mask = (~seg_mask).astype(np.uint8) * 255
    seg_mask = cv2.resize(seg_mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    # Plot using matplotlib
    fig, ax = plt.subplots()
    # Overlay the grayscale image on top of the PNG image
    ax.imshow(HE_img_array, interpolation='none', alpha=1.0)  # Adjust alpha for transparency if needed
    ax.imshow(seg_mask, cmap='gray', interpolation='none', alpha=0.5)
    ax.axis('off')
    plt.show()
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return 

def args():
    parser = argparse.ArgumentParser(description='Segmentation')
    parser.add_argument('--data_type', default='test', choices=['test', 'train'], help='train or test')
    return parser.parse_args()

def segmentation(
    data_type = 'test', # 'train' or 'test'
    visualize = True,
):
    # 1) Dataset Load
    print("Dataset Load")
    dataloaders = get_bci_dataloaders('BCI_dataset', type='segmentation', batch_size=32, num_workers=4, image_size=256)

    # 2) segmentation model load (HookNet)
    print("Load Segmentation Model")
    model = HookNet(encoder_name="resnet18", encoder_weights="imagenet", classes=len(BCSS_CLASSES) + 1)
    checkpoint = torch.load('utils/bcss_fold0_ft_model.pth.tar', map_location="cuda")
    state_dict = remove_module_prefix(checkpoint["state_dict"])
    model.load_state_dict(state_dict)
    model.cuda()
    
    # 3) segmentation mask 생성
    print("Segmentation Inference")
    masks = predict_segmentation(dataloaders[data_type], model)
    numbers = dataloaders[data_type].dataset.numbers
    filenames = dataloaders[data_type].dataset.filenames
    
    #5) segmentation 결과 시각화
    if visualize:
        print("Visualize Segmentation")
        from_dir = dataloaders[data_type].dataset.directory
        to_dir = os.path.join('BCI_dataset', 'segmented_result', data_type)
        assert os.path.exists(from_dir), f'{from_dir} does not exist'
        os.makedirs(to_dir, exist_ok=True)
        for i in range(len(filenames)):
            filename = os.path.join(from_dir, filenames[i])
            savefilename = os.path.join(to_dir, f'result_{numbers[i]}.png')
            visualize_segmentation(masks[i], filename=filename, save_path=savefilename)
            # visualize_segmentation(masks[i], num_classes=len(BCSS_CLASSES), save_path=savefilename)
            print(f"Save: {filename} -> {savefilename}")
        
    # 6) segmentation mask 저장    
    save_to = os.path.join('BCI_dataset', 'segmented_result', data_type, 'masks.pth')
    torch.save(dict([(f'{numbers[i]}', masks[i]) for i in range(len(numbers))]), save_to)
    print(f"Save: {save_to}")
    return 


if __name__ == '__main__':
    args = args()
    random_seed(42)
    segmentation(data_type=args.data_type, visualize=True)