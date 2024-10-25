import shutil
import dsmil as mil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, glob, copy
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
from sklearn.utils import shuffle
from utils import path


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
    Tensor = torch.FloatTensor
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
        if len(feats_list) == 0:
            print('No valid patch extracted from: ' + bags_list[i])
        else:
            df = pd.DataFrame(feats_list)
            bag_dir = os.path.join(save_path, bags_list[i].split(os.path.sep)[-2])
            save_file = os.path.join(bag_dir, bags_list[i].split(os.path.sep)[-1]+'.csv')
            os.makedirs(bag_dir, exist_ok=True)
            df.to_csv(save_file, index=False, float_format='%.4f')
            print('\t: ', save_file)
        
def compute_tree_feats(args, bags_list, embedder_low, embedder_high, save_path=None):
    embedder_low.eval()
    embedder_high.eval()
    num_bags = len(bags_list)
    with torch.no_grad():
        for i in range(0, num_bags): 
            low_patches = get_patch_list(bags_list[i], 'low')
            # low_patches = glob.glob(os.path.join(bags_list[i], '*.jpg')) + glob.glob(os.path.join(bags_list[i], '*.jpeg'))
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
                    for high_patch in high_patches:
                        img = Image.open(high_patch)
                        img = VF.to_tensor(img).float().cuda()
                        feats, classes = embedder_high(img[None, :])
                        
                        if args.tree_fusion == 'fusion':
                            feats = feats.cpu().numpy()+0.25*feats_list[idx]
                        elif args.tree_fusion == 'cat':
                            feats = np.concatenate((feats.cpu().numpy(), feats_list[idx][None, :]), axis=-1)
                        else:
                            raise NotImplementedError(f"{args.tree_fusion} is not an excepted option for --tree_fusion. This argument accepts 2 options: 'fusion' and 'cat'.")
                        
                        feats_tree_list.extend(feats)
                sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i+1, num_bags, idx+1, len(low_patches)))
            if len(feats_tree_list) == 0:
                print('No valid patch extracted from: ' + bags_list[i])
            else:
                df = pd.DataFrame(feats_tree_list)
                bag_dir = os.path.join(save_path, bags_list[i].split(os.path.sep)[-2])
                save_file = os.path.join(bag_dir, bags_list[i].split(os.path.sep)[-1]+'.csv')
                os.makedirs(bag_dir, exist_ok=True)
                df.to_csv(save_file, index=False, float_format='%.4f')
                print('\t: ', save_file)
            # print('\n')            

# def get_abs_path(path): 
#     return os.path.abspath(os.path.join(os.path.dirname(__file__), path)) 

def main():
    parser = argparse.ArgumentParser(description='Compute TCGA features from SimCLR embedder')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size of dataloader [128]')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of threads for datalodaer')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--backbone', default='resnet18', type=str, help='Embedder backbone [resnet18]')
    parser.add_argument('--norm_layer', default='instance', type=str, help='Normalization layer [instance]')
    parser.add_argument('--magnification', default='single', type=str, help='Magnification to compute features. Use `tree` for multiple magnifications. Use `high` if patches are cropped for multiple resolution and only process higher level, `low` for only processing lower level.')
    parser.add_argument('--weights', default=None, type=str, help='Folder of the pretrained weights, simclr/runs/*')
    parser.add_argument('--weights_high', default=None, type=str, help='Folder of the pretrained weights of high magnification, FOLDER < `simclr/runs/[FOLDER]`')
    parser.add_argument('--weights_low', default=None, type=str, help='Folder of the pretrained weights of low magnification, FOLDER <`simclr/runs/[FOLDER]`')
    parser.add_argument('--tree_fusion', default='cat', type=str, help='Fusion method for high and low mag features in a tree method [cat|fusion]')
    parser.add_argument('--dataset', default='TCGA-lung-single', type=str, help='Dataset folder name [TCGA-lung-single]')
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)

    if args.norm_layer == 'instance':
        norm=nn.InstanceNorm2d
        pretrain = False
    elif args.norm_layer == 'batch':  
        norm=nn.BatchNorm2d
        if args.weights == 'ImageNet':
            pretrain = True
        else:
            pretrain = False

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
        
        if args.weights_high == 'ImageNet' or args.weights_low == 'ImageNet' or args.weights== 'ImageNet':
            if args.norm_layer == 'batch':
                print('Use ImageNet features.')
            else:
                raise ValueError('Please use batch normalization for ImageNet feature')
        else:
            # 1) load pretrained weights for high magnification
            for m in ['high', 'low']:
                weight_path = path.get_simclr_chkpt_path(args.weights_high if m=='high' else args.weights_low)
                print(f'Use pretrained features: {weight_path}')
                # get_abs_path(os.path.join('simclr', 'runs', args.weights_high, 'checkpoints', 'model.pth'))
                state_dict_weights = torch.load(weight_path, weights_only=True)
                for i in range(4):
                    state_dict_weights.popitem()
                state_dict_init = i_classifier_h.state_dict() if m=='high' else i_classifier_l.state_dict()
                new_state_dict = OrderedDict()
                for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                    name = k_0
                    new_state_dict[name] = v
                if m=='high':
                    i_classifier_h.load_state_dict(new_state_dict, strict=False)
                else:
                    i_classifier_l.load_state_dict(new_state_dict, strict=False)
                embedder_file = path.get_embedder_path(args.dataset, f'embedder-{m}.pth')
                # os.makedirs(embedder_dir, exist_ok=True)
                torch.save(new_state_dict, embedder_file)

            # # 2) load pretrained weights for low magnification
            # weight_path = path.get_simclr_chkpt_path(args.weights_low) # get_abs_path(os.path.join('simclr', 'runs', args.weights_low, 'checkpoints', 'model.pth'))
            # state_dict_weights = torch.load(weight_path, weights_only=True)
            # for i in range(4):
            #     state_dict_weights.popitem()
            # state_dict_init = i_classifier_l.state_dict()
            # new_state_dict = OrderedDict()
            # for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
            #     name = k_0
            #     new_state_dict[name] = v
            # i_classifier_l.load_state_dict(new_state_dict, strict=False)
            # os.makedirs(get_abs_path(os.path.join('embedder', args.dataset)), exist_ok=True)
            # torch.save(new_state_dict, get_abs_path(os.path.join('embedder', args.dataset, 'embedder-low.pth')))
            


    elif args.magnification == 'single' or args.magnification == 'high' or args.magnification == 'low':  
        i_classifier = mil.IClassifier(resnet, num_feats, output_class=args.num_classes).cuda()

        if args.weights == 'ImageNet':
            if args.norm_layer == 'batch':
                print('Use ImageNet features.')
            else:
                print('Please use batch normalization for ImageNet feature')
        else:
            # if args.weights is not None:
            weight_path = path.get_simclr_chkpt_path(args.weights) 
                # get_abs_path(os.path.join('simclr', 'runs', args.weights, 'checkpoints', 'model.pth'))
            # else:
            #     weight_path = glob.glob('dsmil-wsi/simclr/runs/*/checkpoints/*.pth')[-1]
            state_dict_weights = torch.load(weight_path, weights_only=True)
            for i in range(4):
                state_dict_weights.popitem()
            state_dict_init = i_classifier.state_dict()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
            i_classifier.load_state_dict(new_state_dict, strict=False)
            embedder_file = path.get_embedder_path(args.dataset, 'embedder.pth')
            # os.makedirs(embedder_dir, exist_ok=True)
            torch.save(new_state_dict, embedder_file)
            print(f'Use pretrained features: {weight_path}')
    
    if args.magnification == 'tree' or args.magnification == 'low' or args.magnification == 'high' :
        bags_path = path.get_patch_dir(args.dataset, 'pyramid') + '*/*'
        # bags_path = get_abs_path(os.path.join('..', 'datasets', args.dataset, 'pyramid', '*', '*'))
    else:
        bags_path = path.get_patch_dir(args.dataset, 'single') + '*/*'
        # bags_path = get_abs_path(os.path.join('..', 'datasets', args.dataset, 'single', '*', '*'))
    feats_path = path.get_feature_dir(args.dataset) # get_abs_path(os.path.join('datasets', args.dataset))
    print('bags_path:', bags_path, 'feats_path:', feats_path)
    
    # remove existing folders
    shutil.rmtree(feats_path, ignore_errors=True)    
    os.makedirs(feats_path)
    bags_list = glob.glob(bags_path)
    
    if args.magnification == 'tree':
        compute_tree_feats(args, bags_list, i_classifier_l, i_classifier_h, feats_path)
    else:
        compute_feats(args, bags_list, i_classifier, feats_path, args.magnification)
    class_dirs = sorted(glob.glob(os.path.join(feats_path, '*/'))) # dsml-wsi/datasets/acrobat/*/
    # get_abs_path(os.path.join('datasets', args.dataset, '*'+os.path.sep)))
    # class_dirs = sorted(class_dirs)
    all_df = []
    for i, class_dir in enumerate(class_dirs):
        label = class_dir.split(os.path.sep)[-2]
        bag_csvs = glob.glob(os.path.join(class_dir, '*.csv'))
        bag_df = pd.DataFrame(bag_csvs)
        bag_df['label'] = i 
        csv_file = os.path.join(class_dir, label+'.csv')
        bag_df.to_csv(csv_file, index=False) # dsmil-wsi/datasets/acrobat/{label}/{label}.csv
        all_df.append(bag_df)
    bags_path = pd.concat(all_df, axis=0, ignore_index=True)
    bags_path = shuffle(bags_path)
    csv_file = os.path.join(feats_path, args.dataset+'.csv')
    bags_path.to_csv(csv_file, index=False) # dsmil-wsi/datasets/acrobat/acrobat.csv : aggregated all (csv_file_path, label) pairs
    
if __name__ == '__main__':
    main()