import glob
import os
from skimage import transform, exposure
import tifffile as tiff

from matplotlib import pyplot as plt
import numpy as np
from utils import path
# from utils.path import get_project_root, get_labeled_WSI_files, get_patch_dir, get_raw_patch_file

def image_path():
    return os.path.dirname(os.path.realpath(__file__)) + '/images'

def wsi_path(slide_name, num_class=None) -> str:
    class_name = ['neg', '1', '2', '3']
    num_class = class_name[num_class] if num_class is not None else '*'
    # paths = glob.glob(f"{get_project_root()}/test/output/{slide_name}_*_{num_class}.png")
    return f"{image_path()}/wsi/{slide_name}_{num_class}.png"

def wsi_colormap_path(slide_name) -> str:
    return f"{image_path()}/colormap/{slide_name}_colormap_model1.png"

def attention_map_path(slide_name, pos=None, cropped=True) -> list[str] | str:
    patient, imgtype, type = slide_name.split('_')
    cropped = '_cropped' if cropped else ''
    if pos is None:
        path = glob.glob(f"{image_path()}/attention_map/{patient}_{type}/*_*_*{cropped}.png")
    else:
        path = glob.glob(f"{image_path()}/attention_map/{patient}_{type}/{pos[0]}_{pos[1]}_*{cropped}.png")
        if len(path) == 0:
            print(f"[{slide_name}] Attention map {pos} not found.")
            return None
        path = path[0]
    return path

def patch_path(slide_name, pos=None) -> list[str] | str:
    patient, imgtype, type = slide_name.split('_')
    if pos is None :
        path = glob.glob(f"{image_path()}/patches/{patient}_{type}/*_*.jpeg")
        # path = glob.glob(f'{get_patch_dir("acrobat", f"pyramid_{type}")}/*/{slide_name}/*_*.jpeg')
    else:
        path = f"{image_path()}/patches/{patient}_{type}/{pos[0]}_{pos[1]}.jpeg"
        # path = f'{get_patch_dir("acrobat", f"pyramid_{type}")}/*/{slide_name}/{pos[0]}_{pos[1]}.jpeg'
        # assert os.path.exists(path), f"[{slide_name}] Patch {pos} not found."
    return path

def get_available_patches(slide_name):
    slide_name = slide_name.split('.')[0]
    patient, imgtype, type = slide_name.split('_')
    if type == 'val': type = 'train'
    # patches = patch_path(slide_name)
    patches = attention_map_path(slide_name)
    metadata = [name.split(os.sep)[-1].rstrip('.png').rstrip('_cropped').split('_') for name in patches]
    return patches, metadata

def draw_wsi_colormap(slide_name):
    class_name = ['1', '2', '3', 'neg']
    
    # Read the WSI image 
    wsi_img_path = path.get_raw_WSI_file(slide_name)
    with tiff.TiffFile(wsi_img_path) as tif:
        resolution = 3
        tile_size = (224 * 4) // (1 << resolution)
        wsi_img = tif.pages[resolution].asarray()
        img_shape = wsi_img.shape
    
    # Create a color map for the predicted results  
    color_map_shape = [img_shape[0]//tile_size, img_shape[1]//tile_size, 3]
    print(color_map_shape)
    color_map = np.full(color_map_shape, 255, dtype=np.uint8)
    # colors = [np.array([255, 0, 0]), np.array([0, 255, 0]), np.array([0, 0, 255]), np.array([240, 240, 240]), np.array([10, 10, 10])]
    colors = [
        np.array([175,193,248]), # neg #afc1f8
        np.array([92, 133, 255]), # 1 #5c85ff
        np.array([14, 57, 225]), # 2 #0e39e1
        np.array([6, 36, 157]), # 3 #06249d
        np.array([178,180,184]), # tissue #b2b4b8
    ]
    patches, metadata = get_available_patches(slide_name)
    for meta in metadata:
        col, row, pred = [int(i) for i in meta]
        color_map[row, col] = colors[pred]
        
    # Resize the color map to the WSI image size
    color_map = transform.resize(color_map, (img_shape[0],img_shape[1]), order=0)
    
    fig, ax = plt.subplots()
    # make the background transparent
    color_map = np.dstack((color_map, (color_map.sum(axis=-1) < 255*3).astype(np.uint8) * 255))
    # Add legend for colors
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i]/255, markersize=10, label=class_name[i]) for i in range(len(class_name))]
    ax.legend(handles=legend_elements, loc='lower left', fontsize='large')
    ax.axis('off')
    
    # Save the color map image
    image_path = wsi_colormap_path(slide_name)
    plt.imsave(image_path, color_map.astype(np.uint8))
    plt.close(fig)
    print(f"Color map image saved at : {image_path}")
    return image_path



def geojson_to_png():
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from shapely.affinity import scale

    raw_geojson_files = glob.glob(image_path() + '/segment_raw_103_train/*.geojson')
    output_dir = image_path() + '/segment_103_train'
    for file_path in raw_geojson_files:
        filename = file_path.split('/')[-1].split('.')[0]
        gdf = gpd.read_file(file_path)
        gdf = gdf[gdf['geometry'].type == 'Polygon']
        gdf['geometry'] = gdf['geometry'].apply(lambda geom: scale(geom, yfact=-1, origin=(0, 0)))
        gdf.plot(edgecolor='black', color='#e0e0e0', aspect=1)
        plt.axis('off')
        output_file = f'{output_dir}/{filename}.png'
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())    
        plt.gca().set_position([0, 0, 1, 1])
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=300)
        print(f"Saved : {output_file}")
        plt.close()

def crop_heatmap():
    for slide_name in ['0_HER2_train', '92_HER2_train', '122_HER2_train', '103_HER2_train']:
        heatmap_paths = attention_map_path(slide_name=slide_name, cropped=False)
        for heatmap_path in heatmap_paths:
            cropped_heatmap_path = heatmap_path.replace('.png', '_cropped.png')
            if not os.path.exists(heatmap_path):
                print(f"[{slide_name}] Cropping : ", end=' ')
            else:
                print(f"[{slide_name}] Already Cropped : {cropped_heatmap_path}")
                continue
            with open(heatmap_path, "rb") as image_file:
                heatmap = plt.imread(image_file)
                # Find the bounding box of the non-white area
                non_white_pixels = np.where(np.any(heatmap != [1,1,1,1], axis=-1))
                top_left = np.min(non_white_pixels, axis=1)
                bottom_right = np.max(non_white_pixels, axis=1)
                
                # Crop the heatmap to the bounding box
                cropped_heatmap = heatmap[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
                
                # Save the cropped heatmap
                plt.imsave(cropped_heatmap_path, cropped_heatmap)
                print(f"{cropped_heatmap_path}")
    