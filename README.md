# Breast Cancer Classification

![framework](framework.png)

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
