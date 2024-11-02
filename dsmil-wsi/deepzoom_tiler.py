import csv
import json
from multiprocessing import Process, JoinableQueue
import argparse
import os
import re
import shutil
import sys
import glob
import numpy as np
import math
from unicodedata import normalize
from skimage import io
from skimage.color import rgb2hsv
from skimage.util import img_as_ubyte
from skimage import filters
from PIL import Image, ImageFilter, ImageStat

from utils import path

Image.MAX_IMAGE_PIXELS = None

import openslide
from openslide import open_slide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator

VIEWER_SLIDE_NAME = 'slide'

class TileWorker(Process):
    """A child process that generates and writes tiles."""

    def __init__(self, queue, slidepath, tile_size, overlap, limit_bounds,
                quality, threshold):
        Process.__init__(self, name='TileWorker')
        self.daemon = True
        self._queue = queue
        self._slidepath = slidepath
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._quality = quality
        self._threshold = threshold
        self._slide = None

    def run(self):
        self._slide = open_slide(self._slidepath)
        last_associated = None
        dz = self._get_dz()
        while True:
            data = self._queue.get()
            if data is None:
                self._queue.task_done()
                break
            associated, level, address, outfile = data
            if last_associated != associated:
                dz = self._get_dz(associated)
                last_associated = associated
            try:
                tile = dz.get_tile(level, address)
                edge = tile.filter(ImageFilter.FIND_EDGES)
                edge = ImageStat.Stat(edge).sum
                edge = np.mean(edge)/(self._tile_size**2)
                w, h = tile.size
                if edge > self._threshold:
                    if not (w==self._tile_size and h==self._tile_size):
                        tile = tile.resize((self._tile_size, self._tile_size))
                    tile.save(outfile, quality=self._quality)
            except:
                pass
            self._queue.task_done()
            

    def _get_dz(self, associated=None):
        if associated is not None:
            image = ImageSlide(self._slide.associated_images[associated])
        else:
            image = self._slide
        return DeepZoomGenerator(image, self._tile_size, self._overlap,
                    limit_bounds=self._limit_bounds)


class DeepZoomImageTiler(object):
    """Handles generation of tiles and metadata for a single image."""

    def __init__(self, dz, basename, target_levels, mag_base, format, associated, queue):
        self._dz = dz
        self._basename = basename
        self._format = format
        self._associated = associated
        self._queue = queue
        self._processed = 0
        self._target_levels = target_levels
        self._mag_base = int(mag_base)
        self.cut_row_edge = 3 # cut the edge rows/cols
        self.cut_col_edge = 5

    def run(self):
        self._write_tiles()

    def _write_tiles(self):
        target_levels = [self._dz.level_count-i-1 for i in self._target_levels]
        mag_list = [int(self._mag_base/2**i) for i in self._target_levels]
        mag_idx = 0
        for level in range(self._dz.level_count):
            if not (level in target_levels):
                continue
            tiledir = os.path.join(f"{self._basename}_files", str(mag_list[mag_idx]))
            if not os.path.exists(tiledir):
                os.makedirs(tiledir)
            cols, rows = self._dz.level_tiles[level]
            total = (cols - 2*self.cut_col_edge) * (rows - 2*self.cut_row_edge)
            for row in range(self.cut_row_edge, rows - self.cut_row_edge): # cut the edge
                for col in range(self.cut_col_edge, cols - self.cut_col_edge): # cut the edge
                    tilename = os.path.join(tiledir, f'{col}_{row}.{self._format}')
                    if not os.path.exists(tilename):
                        self._queue.put((self._associated, level, (col, row),
                                    tilename))
                    self._tile_done(total)
            mag_idx += 1

    def _tile_done(self, total):
        self._processed += 1
        count = self._processed
        if count % 100 == 0 or count == total:
            print("Tiling %s: wrote %d/%d tiles" % (
                    self._associated or 'slide', count, total),
                    end='\r', file=sys.stderr)
            if count == total:
                print(file=sys.stderr)


class DeepZoomStaticTiler(object):
    """Handles generation of tiles and metadata for all images in a slide.
    Input:
        slidepath : str, path to the slide image (e.g., 'path/to/WSI/class_name/slide_name.svs')
        basename : str, base folder name for the output files (e.g., 'path/to/WSI_temp')
        mag_levels : tuple, 생성할 타일의 배율 레벨 (e.g., (0, 1))
        base_mag : float, maximum magnification for patch extraction [20]
        objective : float, the default objective power if metadata does not present [20]
        format : str, image format for tiles [jpeg]
        tile_size : int, tile size [224]
        overlap : int, overlap of adjacent tiles [0]
        limit_bounds : bool, limit the bounds of the generated tiles [True]
        quality : int, JPEG compression quality [70]
        workers : int, number of worker processes to start [4]
        threshold : int, threshold for filtering background [15]
    """

    def __init__(self, slidepath, basename, mag_levels, base_mag, objective, format, tile_size, overlap,
                limit_bounds, quality, workers, threshold):
        self._slide = open_slide(slidepath)
        self._basename = basename
        self._format = format
        self._tile_size = tile_size
        self._overlap = overlap
        self._mag_levels = mag_levels
        self._base_mag = base_mag
        self._objective = objective
        self._limit_bounds = limit_bounds
        self._queue = JoinableQueue(2 * workers)
        self._workers = workers
        self._dzi_data = {}
        for _i in range(workers):
            TileWorker(self._queue, slidepath, tile_size, overlap,
                        limit_bounds, quality, threshold).start()

    def run(self):
        self._run_image()
        self._shutdown()

    def _run_image(self, associated=None):
        """Run a single image from self._slide."""
        if associated is None:
            image = self._slide
            basename = self._basename
        else:
            image = ImageSlide(self._slide.associated_images[associated])
            basename = os.path.join(self._basename, self._slugify(associated))
        dz = DeepZoomGenerator(image, self._tile_size, self._overlap,
                    limit_bounds=self._limit_bounds)
        
        MAG_BASE = self._slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
        if MAG_BASE is None:
            MAG_BASE = self._objective
        first_level = int(math.log2(float(MAG_BASE)/self._base_mag)) # raw / input, 40/20=2, 40/40=0
        target_levels = [i+first_level for i in self._mag_levels] # levels start from 0
        target_levels.reverse()
        
        tiler = DeepZoomImageTiler(dz, basename, target_levels, MAG_BASE, self._format, associated,
                    self._queue)
        tiler.run()
        return

    def _url_for(self, associated):
        if associated is None:
            base = VIEWER_SLIDE_NAME
        else:
            base = self._slugify(associated)
        return '%s.dzi' % base

    def _copydir(self, src, dest):
        if not os.path.exists(dest):
            os.makedirs(dest)
        for name in os.listdir(src):
            srcpath = os.path.join(src, name)
            if os.path.isfile(srcpath):
                shutil.copy(srcpath, os.path.join(dest, name))

    @classmethod
    def _slugify(cls, text):
        text = normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode()
        return re.sub('[^a-z0-9]+', '_', text)

    def _shutdown(self):
        for _i in range(self._workers):
            self._queue.put(None)
        self._queue.join()

def nested_patches(img_slide, out_base, level=(0,), ext='jpeg'):
    '''
    Organize patches in a nested structure.
    input:
        img_slide : str, path to the slide image (e.g., 'path/to/WSI/class_name/slide_name.svs')
        out_base : str, path to the output directory (e.g., 'path/to/WSI/class_name')
        level : tuple, levels for patch extraction (e.g., (0, 1))
        ext : str, image format for patches (e.g., 'jpeg')
    '''
    print('\n Organizing patches')
    
    # 이미지 슬라이드의 이름과 클래스를 추출
    img_name = img_slide.split(os.sep)[-1].split('.')[0]
    img_class = img_slide.split(os.sep)[-2]
    # 임시 파일 디렉토리에서 레벨의 개수를 계산
    temp_file_dir = get_abs_path('WSI_temp_files')
    temp_file_subdirs = os.path.join(temp_file_dir, '*')
    n_levels = len(glob.glob(temp_file_subdirs))
    # 패치를 저장할 경로 생성
    bag_path = os.path.join(out_base, img_class, img_name)
    os.makedirs(bag_path, exist_ok=True)
    if len(level)==1: # single level
        patches = glob.glob(os.path.join(temp_file_subdirs, '*.'+ext))
        for i, patch in enumerate(patches):
            patch_name = patch.split(os.sep)[-1]
            shutil.move(patch, os.path.join(bag_path, patch_name))
            sys.stdout.write('\r Patch [%d/%d]' % (i+1, len(patches)))
        print('Done.')
    else: # nested levels
        level_factor = 2**int(level[1]-level[0])
        levels = [int(os.path.basename(i)) for i in glob.glob(temp_file_subdirs)]
        levels.sort()
        low_patches = glob.glob(os.path.join(temp_file_dir, str(levels[0]), '*.'+ext))
        for i, low_patch in enumerate(low_patches):
            low_patch_name = low_patch.split(os.sep)[-1]
            shutil.move(low_patch, os.path.join(bag_path, low_patch_name))
            low_patch_folder = low_patch_name.split('.')[0]
            high_patch_path = os.path.join(bag_path, low_patch_folder)
            os.makedirs(high_patch_path, exist_ok=True)
            low_x = int(low_patch_folder.split('_')[0])
            low_y = int(low_patch_folder.split('_')[1])
            high_x_list = list( range(low_x*level_factor, (low_x+1)*level_factor) )
            high_y_list = list( range(low_y*level_factor, (low_y+1)*level_factor) )
            for x_pos in high_x_list:
                for y_pos in high_y_list:
                    high_patch = glob.glob(os.path.join(temp_file_dir, str(levels[1]), '{}_{}.'.format(x_pos, y_pos)+ext))
                    if len(high_patch)!=0:
                        high_patch = high_patch[0]
                        shutil.move(high_patch, os.path.join(bag_path, low_patch_folder, high_patch.split(os.sep)[-1]))
            try:
                os.rmdir(os.path.join(bag_path, low_patch_folder))
                os.remove(low_patch)
            except:
                pass
            sys.stdout.write('\r Patch [%d/%d]' % (i+1, len(low_patches)))
        print('Done.')
        
def get_abs_path(path):
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_file_directory, path)

def get_label_dict(data):
    assert data == 'acrobat', 'Only acrobat dataset is supported'
    label_csv_file = os.path.join(path.get_data_dir(data), 'meta_data', 'acrobat_label.csv')
    label_dict = {'train': {}, 'valid': {}, 'test': {}}
    with open(label_csv_file, 'r') as f:
        f.readline()  # Skip the header
        reader = csv.reader(f)
        for row in reader:
            patient, dataset, label, _, _ = row
            if label == '0' or label in [None, '', '?']: # 0: negative / 1~3: positive
                label = 'neg' # '0' -> 'neg'
            elif label not in ['0', '1', '2', '3']: # label should be in ['0', '1', '2', '3']
                raise ValueError(f"patient [{patient}] - Invalid label: {label}")
            # classify the dataset into train, valid, test / label into neg, 1, 2, 3
            if dataset == 'test':
                label_dict['test'][patient] = label
            elif dataset == 'valid':
                label_dict['valid'][patient] = label
            else:
                label_dict['train'][patient] = label
    return label_dict

def preprocess_WSI(data, imgtype:str='HER2', type:str='train', slide_format:str='tif')->list:
    assert data == 'acrobat', 'Only acrobat dataset is supported'
    assert imgtype in ['HER2', 'HE', 'PGR', 'KI67', 'ER'], f'Invalid image type {imgtype}'
    assert type in ['train', 'test'], f'Invalid type {type}'
    assert slide_format in ['tif'], f'Invalid slide format {slide_format}'
    label_dict = get_label_dict(data) # {'train': {patient: label}, 'valid': {patient: label}, 'test': {patient: label}}
    label_types = ['neg', '1', '2', '3']
    label_dirs = {i: os.path.join(path.get_data_dir(data), i) for i in label_types}

    # Create label folders
    for label_dir in label_dirs.values():
        os.makedirs(label_dir, exist_ok=True)
    
    # Copy raw WSI files to the label folders
    raw_file_list = path.get_raw_WSI_files(data, imgtype, type, slide_format)
    for file in raw_file_list:
        filename = file.split(os.sep)[-1]
        file_number, img_type, _ = filename.split('_')
        if filename.endswith(f'.{slide_format}') and img_type ==imgtype:
            assert file_number in label_dict[type], f"Patient {file_number} not found in the label dictionary"
            label = label_dict[type][file_number]
            to_file_path = os.path.join(label_dirs[label], filename)
            if os.path.exists(to_file_path):
                print(f"[{filename}] : already exists in {to_file_path}")
            else:
                shutil.copy(file, to_file_path)
                print(f"[{filename}] : copied to {to_file_path}")
    return path.get_labeled_WSI_files(args.dataset, args.imgtype, args.type, args.slide_format)

if __name__ == '__main__':
    Image.MAX_IMAGE_PIXELS = None
    parser = argparse.ArgumentParser(description='Patch extraction for WSI')
    parser.add_argument('-d', '--dataset', type=str, default='acrobat', help='Dataset name') # acrobat
    parser.add_argument('--imgtype', type=str, default='HER2', help='Image type') # HER2
    parser.add_argument('--type', type=str, default='train', choices=['train', 'test'], help='Dataset type [train]') # train
    parser.add_argument('-e', '--overlap', type=int, default=0, help='Overlap of adjacent tiles [0]') # 0
    parser.add_argument('-f', '--format', type=str, default='jpeg', help='Image format for tiles [jpeg]') # jpeg
    parser.add_argument('-v', '--slide_format', type=str, default='tif', help='Image format for tiles [tif]') # tif
    parser.add_argument('-j', '--workers', type=int, default=4, help='Number of worker processes to start [4]') # 4
    parser.add_argument('-q', '--quality', type=int, default=70, help='JPEG compression quality [70]') # 70
    parser.add_argument('-s', '--tile_size', type=int, default=224, help='Tile size [224]') # 224
    parser.add_argument('-b', '--base_mag', type=float, default=20, help='Maximum magnification for patch extraction [20]') # 20
    parser.add_argument('-m', '--magnifications', type=int, nargs='+', default=(1,3), help='Levels for patch extraction [0]') 
    parser.add_argument('-o', '--objective', type=float, default=20, help='The default objective power if metadata does not present [20]')
    parser.add_argument('-t', '--background_t', type=int, default=15, help='Threshold for filtering background [15]')  
    args = parser.parse_args()
    print(f"***Arguments: \n\t{args}")
    levels = tuple(sorted(args.magnifications))
    assert len(levels)<=2, 'Only 1 or 2 magnifications are supported!'
    path_base = path.get_data_dir(args.dataset) 
    out_base = path.get_patch_dir(args.dataset, f'pyramid_{args.type}' if len(levels)==2 else f'single_{args.type}', make=True) 

    # Get all WSI files
    all_slides = path.get_labeled_WSI_files(args.dataset, args.imgtype, args.type, args.slide_format)
    if not len(all_slides):
        print('No slide files found! Preprocess the slides first.')
        all_slides = preprocess_WSI(args.dataset, args.imgtype, args.type, args.slide_format)
    
    shutil.rmtree(get_abs_path('WSI_temp_files'), ignore_errors=True)

    # pos-i_pos-j -> x, y
    for idx, c_slide in enumerate(all_slides):
        save_path = os.path.join(out_base, c_slide.split(os.sep)[-2], c_slide.split(os.sep)[-1].split('.')[0])
        if os.path.exists(save_path):
            print(f'Process slide {idx+1}/{len(all_slides)} : {c_slide} -> {save_path} [Already exists]')
            continue
        print(f'Process slide {idx+1}/{len(all_slides)} : {c_slide} -> {save_path}')
        DeepZoomStaticTiler(c_slide, get_abs_path('WSI_temp'), levels, args.base_mag, args.objective, args.format, args.tile_size, args.overlap, True, args.quality, args.workers, args.background_t).run()
        nested_patches(c_slide, out_base, levels, ext=args.format)
        shutil.rmtree(get_abs_path('WSI_temp_files'), ignore_errors=True)
    print('Patch extraction done for {} slides.'.format(len(all_slides)))