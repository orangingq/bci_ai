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
    - **default setting** : 10x, 2.5x magnification (-m 1 3 -b 20 -d acrobat -v tif)
      ```bash 
      python -m dsmil-wsi.deepzoom_tiler --type=train 
      python -m dsmil-wsi.deepzoom_tiler --type=test 
      ```
    - other options (ex. 10x magnification only)
      ```bash
      python -m dsmil-wsi.deepzoom_tiler -m 1 -b 20 -d acrobat -v tif 
      ```

3. Train a SimCLR embedder.
    
    Move to simclr directory first. (`cd dsmil-wsi/simclr`)
    > `--level=high` for 10x magnification (higher mag.) images, `--level=low` for 2.5x magnification (lower mag.) images

    - Backbone ResNet18, train from scratch
      ```bash
      python run.py --level=high --batch_size=256 --epoch=20 --log_dir=resnet18_scratch_high
      python run.py --level=low --batch_size=256 --epoch=50 --log_dir=resnet18_scratch_low
      ```
    - Backbone ResNet34, train from scratch
      ```bash
      python run.py --level=high --batch_size=128 --epoch=10 --model=resnet34 --log_dir=resnet34_scratch_high
      python run.py --level=low --batch_size=128 --epoch=20 --model=resnet34 --log_dir=resnet34_scratch_low
      ```
    - Backbone ResNet50, train from scratch
      ```bash
      python run.py --level=high --batch_size=128 --epoch=20 --model=resnet50 --log_dir=resnet50_scratch_high
      python run.py --level=low --batch_size=128 --epoch=50 --model=resnet50 --log_dir=resnet50_scratch_low
      ```
    - Backbone ResNet50, use pretrained weights
      ```bash
      python run.py --level=high --batch_size=128 --epoch=20 --model=resnet50 --pretrained --log_dir=resnet50_finetune_high
      python run.py --level=low --batch_size=128 --epoch=50 --model=resnet50 --pretrained --log_dir=resnet50_finetune_low
      ```

4. Compute features using the embedder.

    Move back to bci_ai folder. (`cd ../..`)

- Case 0 : using SimCLR of ResNet18 as backbone & pretrained on ImageNet

  - Final results:
    - Mean Accuracy: 0.48175
    - Mean AUC per Class (1,2,3,neg) = (0.5959, 0.6191, 0.7655, 0.5716)

  ```bash
  python -m dsmil-wsi.compute_feats --weights_low=ImageNet --weights_high=ImageNet --backbone=resnet18 --norm_layer=batch
  ```

- Case 1 : using SimCLR of ResNet50 as backbone & pretrained on ImageNet

  - Final results:
    - Mean Accuracy: 0.5071
    - Mean AUC per Class (1,2,3,neg) = (0.5855, 0.5934, 0.8348, 0.5302)

  ```bash
  python -m dsmil-wsi.compute_feats --weights_low=ImageNet --weights_high=ImageNet --backbone=resnet50 --norm_layer=batch 
  ```

- Case 2 : using SimCLR of ResNet18 as backbone & trained from the scratch on ACROBAT

  - Final results:
    - Mean Accuracy: 0.5637
    - Mean AUC per Class (1,2,3,neg) = (0.6292, 0.7237, 0.8934, 0.7728)

  ```bash
  python -m dsmil-wsi.compute_feats --weights_low=resnet18_scratch_low --weights_high=resnet18_scratch_high 
  ```

- Case 3 : using SimCLR of ResNet50 as backbone & trained from the scratch on ACROBAT

  - Final results:
    - Mean Accuracy:
    - Mean AUC per Class (1,2,3,neg) =

  ```bash
  python -m dsmil-wsi.compute_feats --weights_low=resnet50_scratch_low --weights_high=resnet50_scratch_high --backbone=resnet50 {--norm_layer=batch}
  ```

- Case 4 : using SimCLR of ResNet50 as backbone & finetuned with ACROBAT based on ImageNet pretrained weights

  - Final results:
    - Mean Accuracy:0.46287
    - Mean AUC per Class (1,2,3,neg) = (0.5801, 0.5845, 0.8186, 0.6009)

  ```bash
  python -m dsmil-wsi.compute_feats --weights_low=resnet50_finetune_low --weights_high=resnet50_finetune_high --backbone=resnet50 --norm_layer=batch
  ```

5. Training.

```bash
CUBLAS_WORKSPACE_CONFIG=:16:8 python dsmil-wsi/train_tcga.py --dataset=acrobat --num_classes=4
```
