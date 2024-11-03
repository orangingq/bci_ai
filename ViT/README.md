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

#### 5. Training.

```bash
python -m train --run_name={resnet18_scratch}
```
