# Breast Cancer Classification

![framework](framework.png)

## Download Dataset

Your downloaded dataset should follow below structure:

```
BCI_AI (root directory)
    |--BCI_dataset
        |--HE
            |--train
                |--xxxxx_train_#.png
            |--test
                |--xxxxx_test_#.png
        |--IHC
            |--train
                |--xxxxx_train_#.png
            |--test
                |--xxxxx_test_#.png
```

## Download Pretrained segmentation model weight

```bash
# Download
wget https://github.com/Dylan-H-Wang/msf-wsi/releases/download/v0.1/bcss_fold0_ft_model.pth.tar
# move file into 'utils' directory
mv bcss_fold0_ft_model.pth.tar utils/bcss_fold0_ft_model.pth.tar
```

## How to Run?

```python
CUBLAS_WORKSPACE_CONFIG=:16:8 python -m main --{optional arguments}
```

`CUBLAS_WORKSPACE_CONFIG=:16:8` is added to fix the random seed.

### Options

- `seed :int = 42`
- `dataset :str = 'BCI_dataset'`
- `image_size :int = 224`
- `model_name :str = 'ViT'`
  - one of 'ViT' / 'ResNet18' / 'ResNet34' / 'ResNet50' / 'ResNet101'
- `optimizer_name :str = 'Adam'`
  - one of 'Adam' / 'AdamW'
- `learning_rate :float = 1e-3`
- `log_freq :int = 10`
  - log accuracy every `log_freq` batches
- `load_dir :str = None`
  - default : load pretrained weight
- `save_dir :str = None`
  - default : not save
