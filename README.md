# Breast Cancer Classification

![framework](framework.png)

## Prerequisites

### Git Clone

```bash
# 'bci_ai' git repository 다운로드
git clone https://github.com/orangingq/bci_ai.git
# 다운 받은 bci_ai 폴더로 이동
cd bci_ai
```

### Anaconda Environment

```bash
# 콘다 가상환경 생성 ('bci_ai')
conda env create --file environment.yml
# bci_ai 가상환경 활성화
conda activate bci_ai
```

## Download Dataset

- BCSS dataset : https://drive.google.com/drive/folders/1zqbdkQF8i5cEmZOGmbdQm-EP8dRYtvss?usp=sharing
- ACROBAT : https://snd.se/en/catalogue/dataset/2022-190-1
- BCI dataset : https://github.com/bupt-ai-cz/BCI/blob/main/download_dataset.md

Your downloaded dataset (ACROBAT, BCI dataset, BCSS dataset) should follow below structure:

```
BCI_AI (root directory)
    └── dataset
        ├── acrobat
        │   └── train
        │       └── xxx_{img_type}_train.tif
        │
        └── BCI_dataset
            ├── HE
            │   ├── train
            │   │   └── xxxxx_train_#.png
            │   └── test
            │       └── xxxxx_test_#.png
            └── IHC
                ├── train
                │   └── xxxxx_train_#.png
                └── test
                    └── xxxxx_test_#.png
```

\* BCSS dataset is not supported yet.

## Download Pretrained segmentation model weight

```bash
# Download
wget https://github.com/Dylan-H-Wang/msf-wsi/releases/download/v0.1/bcss_fold0_ft_model.pth.tar
# move file into 'utils' directory
mv bcss_fold0_ft_model.pth.tar BCSS_segmentation/bcss_fold0_ft_model.pth.tar
```

## How to Run? (Patch Classification)

```python
# for CPU-only
python -m main {--optional arguments}

# for CUDA
CUBLAS_WORKSPACE_CONFIG=:16:8 python -m main {--optional arguments}
```

`CUBLAS_WORKSPACE_CONFIG=:16:8` is added to fix the random seed.

#### Ex. To use pretrained (w/o finetuning) model for classification task in CPU

```bash
python -m main --model_name=ViT
```

#### Ex. To finetune the classification model with AdamW optimizer with GPUs

```bash
CUBLAS_WORKSPACE_CONFIG=:16:8 python -m main --finetune --optimizer_name=AdamW
```

### Options

- `seed :int = 42`
- `dataset :str = 'BCI_dataset'`
- `aug_level :int = 0`
  - 0 : no augmentation, 1: simple augmentation (rotation), 2: complex augmentation
- `image_size :int = 224`
- `model_name :str = 'ViT'`
  - one of 'ViT' / 'ResNet18' / 'ResNet34' / 'ResNet50' / 'ResNet101'
- `optimizer_name :str = 'Adam'`
  - one of 'Adam' / 'AdamW'
- `learning_rate :float = 1e-3`
- `log_freq :int = 30`
  - log accuracy every `log_freq` batches
- `finetune :bool = False`
- `load_dir :str = None`
  - default : load pretrained weight
- `save_dir :str = None`
  - default : not save

## How to Run? (WSI Classification using DSMIL)

0. Create and activate Anaconda environment (`dsmil`)

```bash
# 콘다 가상환경 생성 ('dsmil')
conda env create --file dsmil-wsi/env.yml
# dsmil 가상환경 활성화
conda activate dsmil
```

1. Place ACROBAT WSI files as `datasets\acrobat\[CATEGORY_NAME]\[SLIDE_NAME].tif`.
2. Crop patches.

```bash
python dsmil-wsi/deepzoom_tiler.py -m 1 -b 20 -d acrobat -v tif # 10x magnification
python dsmil-wsi/deepzoom_tiler.py -m 1 3 -b 20 -d acrobat -v tif # 10x, 2.5x magnification
```

3. Train an embedder. 

```bash
cd dsmil-wsi/simclr
# train SimCLR for 10x magnification images
CUDA_VISIBLE_DEVICES=0 python run.py --dataset=acrobat --level=high --multiscale=1 --batch_size=256 --epoch=20
# train SimCLR for 2.5x magnification images
CUDA_VISIBLE_DEVICES=1 python run.py --dataset=acrobat --level=low --multiscale=1 --batch_size=256 --epoch=50
```

4. Compute features using the embedder.

```bash
cd ../.. # .../bci_ai
# compute features for 10x magnification
CUDA_VISIBLE_DEVICES=1 python dsmil-wsi/compute_feats.py --dataset=acrobat --num_classes=4 --batch_size=64 --magnification=high
# compute features for 2.5x magnification
CUDA_VISIBLE_DEVICES=1 python dsmil-wsi/compute_feats.py --dataset=acrobat --num_classes=4 --batch_size=64 --magnification=low
# compute multi-scale at once
dsmil-wsi/compute_feats.py --dataset=acrobat --num_classes=4 --weights_low=ImageNet --weights_high=ImageNet --magnification=tree --norm_layer=batch 
```

5. Training.

```bash
CUDA_VISIBLE_DEVICES=2 CUBLAS_WORKSPACE_CONFIG=:16:8 python dsmil-wsi/train_tcga.py --dataset=acrobat --num_classes=4
```
