from simclr import SimCLR
import yaml
from data_aug.dataset_wrapper import DataSetWrapper
import os, glob
import pandas as pd
import argparse

def generate_csv(args):
    if args.level=='high' and args.multiscale==1:
        path_temp = os.path.join('..', 'datasets', args.dataset, 'pyramid_train', '*', '*', '*', '*.jpeg')
        patch_path = glob.glob(path_temp) # /class_name/bag_name/5x_name/*.jpeg
    if args.level=='low' and args.multiscale==1:
        path_temp = os.path.join('..', 'datasets', args.dataset, 'pyramid_train', '*', '*', '*.jpeg')
        patch_path = glob.glob(path_temp) # /class_name/bag_name/*.jpeg
    if args.multiscale==0:
        path_temp = os.path.join('..', 'datasets', args.dataset, 'single_train', '*', '*', '*.jpeg')
        patch_path = glob.glob(path_temp) # /class_name/bag_name/*.jpeg
    print('patch_path:', os.path.abspath(path_temp))
    df = pd.DataFrame(patch_path)
    df.to_csv('all_patches.csv', index=False)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=str, default='low', help='Magnification level to compute embedder (low/high)')
    parser.add_argument('--multiscale', type=int, default=1, help='Whether the patches are cropped from multiscale (0/1-no/yes)')
    parser.add_argument('--dataset', type=str, default='acrobat', help='Dataset folder name')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--log_dir', type=str, default=None, help='Name of the log directory')
    parser.add_argument('--model', type=str, default='resnet18', help='Base model for the encoder')
    parser.add_argument('--pretrained', action='store_true', help='Fine-tune from a pre-trained model')
    args = parser.parse_args()
    # Load config file and dataset
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    if os.environ['CUDA_VISIBLE_DEVICES']:
        config['gpu_ids']=str(os.environ['CUDA_VISIBLE_DEVICES'])
    else:
        config['gpu_ids']='0'
    config['n_gpu']=len(config['gpu_ids'])
    config['batch_size']=args.batch_size
    config['epochs']=args.epochs
    config['model']={'out_dim': 256, 'base_model': args.model, 'pretrained': args.pretrained}
    config['pretrained']=args.pretrained
    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])   
    generate_csv(args)
    simclr = SimCLR(dataset, config, log_dir=args.log_dir)
    simclr.train()


if __name__ == "__main__":
    main()
