import torch
import tifffile as tiff
import argparse, os, glob, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform

from utils import path
from . import dsmil as mil

def draw_img(slide_name, ins_prediction, bag_prediction, pos_arr):
    class_name = ['1', '2', '3', 'neg']
    wsi_img_path = path.get_raw_WSI_file(slide_name)
    with tiff.TiffFile(wsi_img_path) as tif:
        resolution = 3
        tile_size = 224 // (1 << resolution)
        wsi_img = tif.pages[resolution].asarray()
        img_shape = wsi_img.shape
    
    # Create a color map for the predictions  
    color_map_shape = [img_shape[0]//tile_size, img_shape[1]//tile_size, 3]
    color_map = np.zeros(color_map_shape)
    colors = [np.array([255, 0, 0]), np.array([0, 255, 0]), np.array([0, 0, 255]), np.array([255, 255, 255])]
    for pred, pos in zip(ins_prediction, pos_arr):
        color_map[pos[1]-1, pos[0]-1] = colors[pred]
        
    # Overlay the color map on the WSI image
    color_map = transform.resize(color_map, (wsi_img.shape[0],wsi_img.shape[1]), order=0)
    # wsi_img = exposure.rescale_intensity(wsi_img, out_range=(0, 255))

    # Add text to image
    text = f"Prediction: {class_name[bag_prediction]}"
    fig, ax = plt.subplots()
    ax.imshow(wsi_img.astype(np.uint8), alpha=0.4)
    ax.imshow(color_map.astype(np.uint8), alpha=0.6)
    text_x, text_y = 30, 30
    ax.text(text_x, text_y, text, fontsize=15, color='white', ha='left', va='top')
    # Add legend for colors
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i]/255, markersize=10, label=class_name[i]) for i in range(len(class_name))]
    ax.legend(handles=legend_elements, loc='lower left', fontsize='large')
    ax.axis('off')
    
    # Save the image
    image_path = path.get_test_path('output') + f'/{slide_name}.png'
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Image saved at : {image_path}")
    return


def get_feats_label(slide_name, checkpoint, num_classes):
    '''Get the bag features and label of the given slide_name from the csv file'''
    dataset = 'acrobat'
    patient, imgtype, type = slide_name.split('_')
    if type == 'val': type = 'train'
    bags_csv = os.path.join(path.get_feature_dir(dataset, checkpoint, type), f'{dataset}.csv') # '.../acrobat.csv'

    df = pd.read_csv(bags_csv)
    slide_name_row = df[df['0'].str.contains('/'+slide_name+'.csv')]
    if slide_name_row.empty:
        raise ValueError(f"Slide name {slide_name} not found in the CSV file.")
    feats_csv_path = slide_name_row['0'].iloc[0]
    df = pd.read_csv(feats_csv_path)
    feats = df.reset_index(drop=True) # do not shuffle the order of the patches
    feats = torch.tensor(feats.to_numpy(), dtype=torch.float32)
    label = np.zeros(num_classes)
    if num_classes==1:
        label[0] = slide_name_row['label'].iloc[0]
    else:
        n_labels = int(slide_name_row['label'].iloc[0])
        if n_labels<=(len(label)-1): 
            label[n_labels] = 1
        
    return feats, label


def init_model(args):
    '''Initialize the model with the pretrained weights'''
    # features are already extracted, so only need FC layer for instance classifier
    i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
    b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=args.dropout_node, nonlinear=args.non_linearity).cuda()
    milnet = mil.MILNet(i_classifier, b_classifier).cuda()
    
    # Load the pretrained model
    model_ckpt = glob.glob(os.path.join(path.get_test_path(args.checkpoint, make=False), '*.pth'))[-1]
    thres_file_name = model_ckpt.split(os.sep)[-1].split('.')[0].replace('weights', 'threshold')
    thres_ckpt = os.path.join(path.get_test_path(args.checkpoint, make=False), f'{thres_file_name}.json')
    assert os.path.exists(thres_ckpt), f"Thresholds file not found for {thres_file_name}"
    state_dict = torch.load(model_ckpt, weights_only=False)
    milnet.load_state_dict(state_dict)
    thresholds_optimal = json.load(open(thres_ckpt))
    return milnet, thresholds_optimal

def main():
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    parser.add_argument('--slide_name', type=str, help='Slide name to run the inference on')
    parser.add_argument('--num_classes', default=4, type=int, help='Number of output classes [4]')
    parser.add_argument('--lr', default=0.0001, type=float, help='Initial learning rate [0.0001]')
    parser.add_argument('--num_epochs', default=50, type=int, help='Number of total training epochs [100]')
    parser.add_argument('--stop_epochs', default=10, type=int, help='Skip remaining epochs if training has not improved after N epochs [10]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight decay [1e-3]')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint run name')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [1]')
    parser.add_argument('--average', type=bool, default=False, help='Average the score of max-pooling and bag aggregating')

    args = parser.parse_args()
    args.dataset = 'acrobat'
    label_name = ['1', '2', '3', 'neg']

    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    
    feats, label = get_feats_label(args.slide_name, args.checkpoint, args.num_classes)
    label = np.array(label, dtype=int).argmax()

    assert label < args.num_classes, f"label = {label} >= args.num_classes = {args.num_classes}"
    label = label_name[label]
    patient, imgtype, type = args.slide_name.split('_')
    if type == 'val': type = 'train'
    dir = os.path.join(path.get_patch_dir(type=f'pyramid_{type}'), label, args.slide_name.split(os.sep)[-1].split('.')[0])
    pos_arr = [name.split(os.sep)[-1].rstrip('.jpeg').split('_') for name in glob.glob(dir + '/*/*.jpeg')]
    pos_arr  = [[int(row), int(col)] for [row, col] in pos_arr]
    assert len(pos_arr) == feats.shape[0], f"Number of patches in {dir} = {len(pos_arr)} != number of patches in feats = {feats.shape[0]}"
    
    args.feats_size = feats.shape[1]
    milnet, thresholds_optimal = init_model(args)
    milnet.eval()
    with torch.no_grad():
        assert not torch.isnan(feats).any(), f"bag_feats contains {torch.isnan(feats).sum()} NaN elements"
        ins_prediction, bag_prediction, A, _ = milnet(feats.cuda())
    
    ins_prediction = torch.sigmoid(ins_prediction).squeeze().cpu().numpy()
    bag_prediction = torch.sigmoid(bag_prediction).squeeze().cpu().numpy()
    
    ins_prediction, bag_prediction = ins_prediction.argmax(axis=1), bag_prediction.argmax()
    print(f"Slide {args.slide_name} is detected as: {label_name[bag_prediction]} (True label: {label})")

    assert ins_prediction.max() < args.num_classes, f"ins_prediction = {ins_prediction.max()} >= args.num_classes = {args.num_classes}"
    assert bag_prediction < args.num_classes, f"bag_prediction = {bag_prediction} >= args.num_classes = {args.num_classes}"

    draw_img(args.slide_name, ins_prediction, bag_prediction, pos_arr)
    #TODO : A ? 
    return
    

if __name__ == '__main__':
    main()