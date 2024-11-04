import dsmil as mil
from utils import path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF

import sys, argparse, os, glob, copy
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
from sklearn.utils import shuffle


class BagDataset():
    def __init__(self, csv_file, transform=None):
        self.files_list = csv_file
        self.transform = transform
    def __len__(self):
        return len(self.files_list)
    def __getitem__(self, idx):
        temp_path = self.files_list[idx]
        img = os.path.join(temp_path)
        img = Image.open(img)
        sample = {'input': img}
        
        if self.transform:
            sample = self.transform(sample)
        return sample 

class ToTensor(object):
    def __call__(self, sample):
        img = sample['input']
        img = VF.to_tensor(img)
        return {'input': img} 
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

def bag_dataset(args, csv_file_path):
    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                    transform=Compose([
                                        ToTensor()
                                    ]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)

def get_patch_list(bag, magnification):
    if magnification=='single' or magnification=='low':
        patch_list = glob.glob(os.path.join(bag, '*.jpg')) + glob.glob(os.path.join(bag, '*.jpeg'))
    elif magnification=='high':
        patch_list = glob.glob(os.path.join(bag, '*'+os.sep+'*.jpg')) + glob.glob(os.path.join(bag, '*'+os.sep+'*.jpeg'))
    return patch_list

def compute_feats(args, bags_list, i_classifier, save_path=None, magnification='single'):
    i_classifier.eval()
    num_bags = len(bags_list)
    for i in range(0, num_bags):
        feats_list = []
        patch_list = get_patch_list(bags_list[i], magnification)
        dataloader, bag_size = bag_dataset(args, patch_list)
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().cuda() 
                if torch.isnan(patches).any():
                    print(f"iteration : {iteration}, {torch.isnan(patches).sum()} NAN elements")
                    continue
                feats, classes = i_classifier(patches)
                feats = feats.cpu().numpy()
                assert not np.isnan(feats).any(), f"{iteration} feats contains NaN elements"
                feats_list.extend(feats)
                sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i+1, num_bags, iteration+1, len(dataloader)))
        save_feats(bags_list[i], feats_list, save_path)
    return

def compute_tree_feats(args, bags_list, embedder_low, embedder_high, save_path=None):
    embedder_low.eval()
    embedder_high.eval()
    num_bags = len(bags_list)
    with torch.no_grad():
        for i in range(0, num_bags): 
            low_patches = get_patch_list(bags_list[i], 'low')
            feats_list = []
            feats_tree_list = []
            dataloader, bag_size = bag_dataset(args, low_patches)
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().cuda()
                feats, classes = embedder_low(patches)
                feats = feats.cpu().numpy()
                assert not np.isnan(feats).any(), f"feats contains {np.isnan(feats).sum()} NaN elements"
                feats_list.extend(feats)
            for idx, low_patch in enumerate(low_patches): # for each low patch, find corresponding high patches
                high_folder = os.path.dirname(low_patch) + os.sep + os.path.splitext(os.path.basename(low_patch))[0]
                high_patches = glob.glob(high_folder+os.sep+'*.jpg') + glob.glob(high_folder+os.sep+'*.jpeg')
                if len(high_patches) == 0:
                    pass
                else:
                    imgs = [VF.to_tensor(Image.open(high_patch)) for high_patch in high_patches]
                    # img = VF.to_tensor(img).float().cuda()
                    imgs = torch.stack(imgs).float().cuda()
                    feats, classes = embedder_high(imgs)
                    if args.tree_fusion == 'fusion':
                        feats = feats.cpu().numpy()+0.25*feats_list[idx]
                    elif args.tree_fusion == 'cat':
                        feats = np.concatenate((feats.cpu().numpy(), np.tile(feats_list[idx], (feats.size(0), 1))), axis=1)
                    else:
                        raise NotImplementedError(f"{args.tree_fusion} is not an excepted option for --tree_fusion. This argument accepts 2 options: 'fusion' and 'cat'.")
                        
                    feats_tree_list.extend(feats)
                sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i+1, num_bags, idx+1, len(low_patches)))
            save_feats(bags_list[i], feats_tree_list, save_path)
    return

def save_feats(bag_name, feats_list, save_path):
    if len(feats_list) == 0:
        print('No valid patch extracted from: ' + bag_name)
    else:
        df = pd.DataFrame(feats_list)
        bag_dir = os.path.join(save_path, bag_name.split(os.path.sep)[-2])
        save_file = os.path.join(bag_dir, bag_name.split(os.path.sep)[-1]+'.csv')
        os.makedirs(bag_dir, exist_ok=True)
        df.to_csv(save_file, index=False, float_format='%.4f')
        print('\t: ', save_file)
    return

def loosely_load_weights(model, weight_path, save_file=None):
    '''loosely load weights from pretrained model, except for the last 4 layers'''
    state_dict_weights = torch.load(weight_path, weights_only=True)
    for i in range(4):
        state_dict_weights.popitem()
    state_dict_init = model.state_dict()
    new_state_dict = OrderedDict()
    for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
        name = k_0
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    
    if save_file is not None: # save the embedder weights
        torch.save(new_state_dict, save_file)
    return model

def build_iclassifier(args):
    '''Build the instance classifier(s) with pretrained weights'''
    # set the normalization layer type
    if args.norm_layer == 'instance':
        norm=nn.InstanceNorm2d
        pretrain = False
    elif args.norm_layer == 'batch':  
        norm=nn.BatchNorm2d
        if args.weights_high == 'ImageNet' or args.weights_low == 'ImageNet':
            pretrain = True
        else:
            pretrain = False

    # set the backbone resnet model
    if args.backbone == 'resnet18':
        resnet = models.resnet18(weights="IMAGENET1K_V1" if pretrain else None, norm_layer=norm)
        num_feats = 512
    if args.backbone == 'resnet34':
        resnet = models.resnet34(weights="IMAGENET1K_V1" if pretrain else None, norm_layer=norm)
        num_feats = 512
    if args.backbone == 'resnet50':
        resnet = models.resnet50(weights="IMAGENET1K_V2" if pretrain else None, norm_layer=norm)
        num_feats = 2048
    if args.backbone == 'resnet101':
        resnet = models.resnet101(weights="IMAGENET1K_V2" if pretrain else None, norm_layer=norm)
        num_feats = 2048
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Identity()
    
    if args.magnification == 'tree' and args.weights_high != None and args.weights_low != None:
        i_classifier_h = mil.IClassifier(resnet, num_feats, output_class=args.num_classes).cuda()
        i_classifier_l = mil.IClassifier(copy.deepcopy(resnet), num_feats, output_class=args.num_classes).cuda()
        
        if args.weights_high == 'ImageNet' or args.weights_low == 'ImageNet':
            if args.norm_layer == 'batch':
                print('Use ImageNet features.')
            else:
                raise ValueError('Please use batch normalization for ImageNet feature')
        else:
            # load pretrained weights for high & low magnification
            for m in ['high', 'low']:
                weight_path = path.get_simclr_chkpt_path(args.weights_high if m=='high' else args.weights_low)
                print(f'Use pretrained features: {weight_path}')
                save_file = path.get_embedder_path(args.dataset, f'embedder-{m}.pth')
                if m == 'high':
                    i_classifier_h = loosely_load_weights(i_classifier_h, weight_path, save_file)
                else:
                    i_classifier_l = loosely_load_weights(i_classifier_l, weight_path, save_file)
        return i_classifier_l, i_classifier_h

    elif args.magnification == 'single' or args.magnification == 'high' or args.magnification == 'low':  
        i_classifier = mil.IClassifier(resnet, num_feats, output_class=args.num_classes).cuda()

        if args.weights_low == 'ImageNet':
            if args.norm_layer == 'batch':
                print('Use ImageNet features.')
            else:
                print('Please use batch normalization for ImageNet feature')
        else:
            weight_path = path.get_simclr_chkpt_path(args.weights_low) 
            print(f'Use pretrained features: {weight_path}')
            save_file = path.get_embedder_path(args.dataset, 'embedder.pth')
            i_classifier = loosely_load_weights(i_classifier, weight_path, save_file)
        return i_classifier


def main():
    parser = argparse.ArgumentParser(description='Compute ACROBAT features from SimCLR embedder')
    parser.add_argument('--num_classes', default=4, type=int, help='Number of output classes [4]')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size of dataloader [128]')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of threads for datalodaer')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--backbone', default='resnet18', type=str, help='Embedder backbone [resnet18]')
    parser.add_argument('--norm_layer', default='instance', type=str, help='Normalization layer [instance]')
    parser.add_argument('--magnification', default='tree', type=str, help='Magnification to compute features. Use `tree` for multiple magnifications. Use `high` if patches are cropped for multiple resolution and only process higher level, `low` for only processing lower level.')
    parser.add_argument('--weights_high', default=None, type=str, help='Folder of the pretrained weights of high magnification, FOLDER < `simclr/runs/[FOLDER]`')
    parser.add_argument('--weights_low', default=None, type=str, help='Folder of the pretrained weights of low magnification, FOLDER <`simclr/runs/[FOLDER]`')
    parser.add_argument('--tree_fusion', default='cat', type=str, help='Fusion method for high and low mag features in a tree method [cat|fusion]')
    parser.add_argument('--dataset', default='acrobat', type=str, help='Dataset folder name [acrobat]')
    parser.add_argument('--type', default='train', type=str, choices=['train', 'test'], help='Type of the dataset [train]')
    parser.add_argument('--run_name', default=None, type=str, help='Run name of the experiment')
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    
    if args.magnification == 'tree' and args.weights_high != None and args.weights_low != None:
        i_classifier_l, i_classifier_h = build_iclassifier(args)
    elif args.magnification == 'single' or args.magnification == 'high' or args.magnification == 'low':  
        i_classifier = build_iclassifier(args)
    
    if args.magnification == 'tree' or args.magnification == 'low' or args.magnification == 'high' :
        bags_path = path.get_patch_dir(args.dataset, f'pyramid_{args.type}') + '/*/*'
    else:
        bags_path = path.get_patch_dir(args.dataset, f'single_{args.type}') + '/*/*'
    if args.run_name is None:
        args.run_name = args.weights_high.split("_high")[0]
    feats_path = path.get_feature_dir(args.dataset, run_name=args.run_name, type=args.type, exists=False)
    print('run_name:', args.run_name, '\nbags_path:', bags_path, 'feats_path:', feats_path)
     
    bags_list = glob.glob(bags_path)
    
    if args.magnification == 'tree':
        compute_tree_feats(args, bags_list, i_classifier_l, i_classifier_h, save_path=feats_path)
    else:
        compute_feats(args, bags_list, i_classifier, feats_path, args.magnification)
    class_dirs = sorted(glob.glob(os.path.join(feats_path, '*/'))) # datasets/acrobat/features/{run_name}/*/
    all_df = []
    for i, class_dir in enumerate(class_dirs):
        label = class_dir.split(os.path.sep)[-2]
        bag_csvs = glob.glob(os.path.join(class_dir, '*.csv'))
        bag_df = pd.DataFrame(bag_csvs)
        bag_df['label'] = i 
        csv_file = os.path.join(feats_path, label+'.csv')
        bag_df.to_csv(csv_file, index=False) # datasets/acrobat/features/{run_name}/{label}.csv
        all_df.append(bag_df)
    bags_path = pd.concat(all_df, axis=0, ignore_index=True)
    bags_path = shuffle(bags_path)
    csv_file = os.path.join(feats_path, args.dataset+'.csv')
    bags_path.to_csv(csv_file, index=False) # datasets/acrobat/features/{run_name}/acrobat.csv : aggregated all (csv_file_path, label) pairs
    
if __name__ == '__main__':
    main()