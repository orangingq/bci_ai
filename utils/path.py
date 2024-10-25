import glob
import os

def get_project_root() -> str: # 'bci_ai'
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_data_dir(data:str='acrobat') -> str:
    '''
    data: 'acrobat'
    '''
    path = os.path.join(get_project_root(), 'datasets', data)
    assert os.path.exists(path), f'Dataset Path {path} does not exist'
    return path

def get_WSI_dir(data:str='acrobat', type:str='train') -> str:
    '''
    data: 'acrobat'
    type: one of 'train', 'valid', 'test'
    '''
    path = os.path.join(get_data_dir(data), type, 'WSI')
    assert os.path.exists(path), f'WSI Path {path} does not exist'
    return path

def get_patch_dir(data:str='acrobat', type:str='pyramid') -> str:
    '''
    data: 'acrobat'
    type: one of 'pyramid', 'single', 'pyramid_HE'
    '''
    path = os.path.join(get_data_dir(data), type)
    assert os.path.exists(path), f'Patch Path {path} does not exist'
    return path

### dsml-wsi
def get_dsmil_root() -> str: # 'dsmil-wsi'
    return os.path.join(get_project_root(), 'dsmil-wsi')

def get_feature_dir(data:str='acrobat') -> str:
    '''
    data: 'acrobat'
    '''
    path = os.path.join(get_dsmil_root(), 'datasets', data)
    assert os.path.exists(path), f'Feature Path {path} does not exist'
    return path

def get_simclr_chkpt_path(run_name:str=None) -> str:
    '''run_name: ex. 'Oct14_22-27-55_server' '''
    if run_name:
        path = os.path.join(get_dsmil_root(), 'simclr', 'runs', run_name, 'checkpoints', 'model.pth')
    else:
        path = glob.glob(os.path.join(get_dsmil_root(), 'simclr', 'runs', '*', 'checkpoints', '*.pth'))[-1]
    assert os.path.exists(path), f'SimCLR Checkpoint Path {path} does not exist'
    return path

def get_embedder_path(data:str='acrobat', filename:str=None) -> str:
    '''
    data='acrobat' 
    Return path to the embedder checkpoint. If filename is None, return the directory path.
    '''
    path = os.path.join(get_dsmil_root(), 'embedder', data)
    os.makedirs(path, exist_ok=True)
    if filename:
        path = os.path.join(get_dsmil_root(), 'embedder', data, filename)
        assert os.path.exists(path), f'Embedder Checkpoint Path {path} does not exist'
    return path
