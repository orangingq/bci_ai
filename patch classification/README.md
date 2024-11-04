# Breaset Cancer HER2 IHC Patch Classification

## Download Dataset

- BCI : https://bupt-ai-cz.github.io/BCI/
- ACROBAT : https://snd.se/en/catalogue/dataset/2022-190-1


#### Your downloaded dataset should follow below structure:

```
BCI_AI (root directory)
    └── dataset
        ├── bci
        │    ├── train 
        │    │   └── xxx_train.png or jpeg
        │    └── test 
        │        └──xxx_test.png or jpeg
        ├── acrobat_patch
        │    ├── 0
        │    ├── 1 
        │    │   └── xxx.png or jpeg
        │    └── 3 
        │        └──xxx.png or jpeg
        ├── BCI_train_label.csv
        └── BCI_test_label.csv
```

## How to Run?

#### 1. Training

```bash
python vit.py   --lr=1e-5
                --root_dir='/pathology/'
                --augmentation_level=3  # 0, 1, 2, 3
```

#### 2. Evaluation

```bash
python evaluate.py
```

