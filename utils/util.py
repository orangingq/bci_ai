from collections import OrderedDict
import os
import numpy as np
import torch
import random
from PIL import Image
from matplotlib import pyplot as plt

def random_seed(seed):
    '''set random seed'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return


def show_image(tensor, title=''):
    '''Show image from tensor'''
    array = tensor.numpy()
    array = np.transpose(array, (1, 2, 0))
    plt.imshow(array)
    plt.title(title)
    plt.axis('off')
    plt.show()
    return


def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[len('module.'):]  # 'module.' prefix 제거
        new_state_dict[k] = v
    return new_state_dict