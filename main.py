import os
import torch
from BCSS_segmentation import segmentation
import utils.args as args
from utils.args import get_model
from utils import random_seed, load_checkpoint
from classification import finetune_classification, inference
from datasets.BCI_dataset.dataloader import get_bci_dataloaders

def compute_final_grade(grades, tumor_portion):
    final_grades = [] #TODO: compute final grade
    return final_grades

def framework():
    # 1. fine-tuning classification model
    if args.finetune:
        print("Fine-tuning classification model")
        dataloaders, model1 = finetune_classification()
    else:
        print("Skip Fine-tuning classification model")
        dataloaders = get_bci_dataloaders(args.dataset, type='classification', batch_size=32, num_workers=4, image_size=args.image_size)
        model1 = load_checkpoint(get_model())[0]
        model1.cuda()
    
    # 2. inference
    print("Classification Inference")
    data_type = 'test'
    grades = inference(dataloaders[data_type], model1)
    
    # 3. Load Segmentation Result
    load_mask_from = os.path.join('BCI_dataset', 'segmented_result', data_type, 'masks.pth')
    if not os.path.exists(load_mask_from):
        print(f"{load_mask_from} does not exist. Run segmentation first.")
        segmentation(data_type=data_type, visualize=False)
    mask_dict = torch.load(f'BCI_dataset/segmented_result/{data_type}/masks.pth')
    tumor_portion = [mask_dict[num].sum()/mask_dict[num].size for num in dataloaders[data_type].dataset.numbers]
    
    # 4. compute final grade
    print("Compute Final Grade")
    final_grades = compute_final_grade(grades, tumor_portion) #TODO: compute final grade
    
    return final_grades


if __name__ == '__main__':
    args.set_args()
    random_seed(args.seed)
    framework()