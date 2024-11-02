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


def get_raw_WSI_file(slide_name:str, label:str=None, slide_format:str='tif'):
    '''get raw (unlabeled) WSI files'''
    patient, imgtype, type = slide_name.split('_')
    if type == 'val': type = 'train'
    if label is None: label = '*'
    file_path = f'{get_data_dir("acrobat")}/{label}/{slide_name}.{slide_format}'
    file = glob.glob(file_path)
    assert len(file) == 1, f'retrieved file : {file}'
    return file[0]

def get_raw_WSI_files(data:str='acrobat', imgtype:str='HER2', type:str='train', slide_format:str='tif') -> list:
    '''get raw (unlabeled) WSI files'''
    assert data in ['acrobat'], f'Invalid data {data}'
    assert imgtype in ['HER2', 'HE'], f'Invalid imgtype {imgtype}'
    assert type in ['train', 'test'], f'Invalid type {type}'
    assert slide_format in ['tif'], f'Invalid slide_format {slide_format}'
    file_path = os.path.join(get_data_dir(data), type, 'WSI', f'*_{imgtype}_{type}.{slide_format}')
    valid_path = os.path.join(get_data_dir(data), 'valid', 'WSI', f'*_{imgtype}_val.{slide_format}')
    raw_files = glob.glob(file_path) + (glob.glob(valid_path) if type == 'train' else [])
    assert len(raw_files) > 0, f'No raw WSI files found in {file_path}'
    return raw_files
    
def get_labeled_WSI_files(data:str='acrobat', imgtype:str='HER2', type:str='train', slide_format:str='tif') -> list:
    '''get labeled WSI files'''
    if data == 'acrobat':
        path_base = get_data_dir(data)
        file_path = os.path.join(path_base, '*', f'*_{imgtype}_{type}.{slide_format}') # ex) 'datasets/acrobat/{class}/{patient}_{imgtype}_train.tif'
        valid_path = os.path.join(path_base, 'valid', f'*_{imgtype}_val.{slide_format}')
        return glob.glob(file_path) + (glob.glob(valid_path) if type == 'train' else [])
    else:
        file_path = os.path.join(path_base, '*/*.'+slide_format)
        return glob.glob(file_path) 

def get_WSI_dir(data:str='acrobat', type:str='train') -> str:
    '''
    data: 'acrobat'
    type: one of 'train', 'valid', 'test'
    '''
    path = os.path.join(get_data_dir(data), type, 'WSI')
    assert os.path.exists(path), f'WSI Path {path} does not exist'
    return path

def get_patch_dir(data:str='acrobat', type:str='pyramid', make:bool=False) -> str:
    '''
    data: 'acrobat'
    type: starting with 'pyramid' or 'single'. ex) 'pyramid_HE', 'pyramid_train' are also valid.
    '''
    path = os.path.join(get_data_dir(data), type)
    if make and not os.path.exists(path):
        os.makedirs(path)
    assert os.path.exists(path), f'Patch Path {path} does not exist'
    return path

### dsml-wsi
def get_dsmil_root() -> str: # 'dsmil-wsi'
    '''Return the root directory of dsmil-wsi (.../dsmil-wsi)'''
    return os.path.join(get_project_root(), 'dsmil-wsi')

def get_feature_dir(data:str='acrobat', run_name:str='', type:str='train', exists:bool=True, make:bool=True) -> str:
    '''
    data: 'acrobat'
    type: 'train' or 'test'
    exists: True if the directory must exist. False if the directory should not exist.
    '''
    assert type in ['train', 'test'], f'Invalid type {type}'
    if data == 'acrobat':
        assert len(run_name) > 0, f'Specify run_name {run_name}'
        path = os.path.join(get_dsmil_root(), 'datasets', data, run_name, type)
    else:
        path = os.path.join(get_dsmil_root(), 'datasets', data)
    assert os.path.exists(path) == exists, f'Feature Path {path}{" does not" if exists else ""} exist'
    if make and not os.path.exists(path):
        os.makedirs(path)
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

def get_test_path(run_name:str=None, make=True)->str:
    '''
    Return the test directory path. If run_name is None, return the root test directory path.
    run_name: ex. 'resnet18_finetune'
    '''
    path = os.path.join(get_dsmil_root(), 'test')
    if run_name is not None:
        if make and not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, run_name)
    if make and not os.path.exists(path):
        os.makedirs(path)
    assert os.path.exists(path), f'Test Path {path} does not exist'
    return path