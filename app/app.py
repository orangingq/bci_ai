import base64
import glob
import math
from flask import Flask, abort, jsonify, render_template, request, send_file
import subprocess
import os
from app import utility as util
# from app.utility import draw_wsi_colormap, image_path, wsi_colormap_path, wsi_path
from utils.path import get_project_root, get_labeled_WSI_files, get_patch_dir, get_raw_patch_file
import numpy as np
from PIL import Image

app = Flask(__name__)
REPRESENTATIVE_SLIDES = {0 : "92_HER2_train", 1 : "0_HER2_train", 2 : "122_HER2_train", 3 : "103_HER2_train"}
SLIDES_SIZE = {0: [46, 24], 1:[52, 30], 2: [52, 20], 3:[46, 25]}
# SLIDES_SIZE = {0: [46, 24], 1:[52, 30], 2: [47, 17], 3:[46, 25]}

@app.route('/')
def index():
    path = os.path.join('index.html')
    return render_template(path)

@app.route('/get_wsi', methods=['POST'])
def get_WSI():
    '''Get the WSI image and colormap for the given class'''
    num_class = request.get_json().get('class')
    slide_name = REPRESENTATIVE_SLIDES[num_class]

    # Check if slide exists
    # available_slides = get_labeled_WSI_files(type='train') + get_labeled_WSI_files(type='val') + get_labeled_WSI_files(type='test')
    # available_slides = [slide.split('/')[-1].split('.')[0] for slide in available_slides]
    # if slide_name not in available_slides:
    #     abort(404, description=f"Slide {slide_name} not found.") # Available slides: {available_slides}

    WSI_path = util.wsi_path(slide_name)
    colormap_path = util.wsi_colormap_path(slide_name)
    
    # Run inference if the image does not exist
    if len(glob.glob(WSI_path)) != 1:
        result = run_inference(slide_name)
        if result.returncode != 0:
            abort(500, description=f"Error running inference: {result.stderr}")
            return result.stderr, 500
    WSI_path = glob.glob(WSI_path)[0]

    colormap_path = util.draw_wsi_colormap(slide_name)

    detected_label = int(WSI_path.split('_')[-1].split('.')[0].replace('neg', '0'))
    true_label = num_class
    # data payload
    with open(WSI_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    with open(colormap_path, "rb") as image_file:
        encoded_colormap = base64.b64encode(image_file.read()).decode('utf-8')
    message = {"detected": detected_label, 
                "true": true_label, 
                "image": encoded_image,
                "colormap": encoded_colormap,
                }
    return jsonify(message)


@app.route('/get_patch', methods=['POST'])
def get_patch():
    class_name = ['neg', '1', '2', '3']
    num_class = request.get_json().get('class')
    slide_name = REPRESENTATIVE_SLIDES[num_class]
    x, y = request.get_json().get('x'), request.get_json().get('y')
    print(f"clicked: x {x} , y {y}")
    pos = SLIDES_SIZE[num_class]
    pos = [math.ceil(x*pos[0])-1, math.ceil(y*pos[1])-1]
    print(f"pos: col {pos[0]} , row {pos[1]}")
    patient, imgtype, type = slide_name.split('_')
    patch_dict = {
        "slide": slide_name,
        "pos": pos,
        "patch": None,
        "segmented": None,
        "attention": None,
        "label": None,
    }
    patch_file = get_raw_patch_file(slide_name, label=class_name[num_class], pos=pos)
    if patch_file is None:
        abort(404, description=f"Patch {pos} not found.")
    
    with open(patch_file, "rb") as image_file:
            patch_dict['patch'] = base64.b64encode(image_file.read()).decode('utf-8')
        
    segmented = f"{util.image_path()}/segment_{patient}_{type}/{pos[0]}_{pos[1]}.png"
    attention = util.attention_map_path(slide_name, pos)
    if attention is not None:
        # data payload
        with open(attention, "rb") as image_file:
            patch_dict['attention'] = base64.b64encode(image_file.read()).decode('utf-8')
        patch_dict['label'] = attention.split('_')[-1].split('.')[0]
    else:
        print(f"Attention map {pos} not found.")
    if os.path.exists(segmented):
        with open(segmented, "rb") as image_file:
            patch_dict['segmented'] = base64.b64encode(image_file.read()).decode('utf-8')
    else:
        print(f"Segmented image {segmented} not found.")
    return patch_dict


def run_inference(slide_name):
    '''Run inference code of WSI classification model'''
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{get_project_root()}"
    env['CUDA_VISIBLE_DEVICES'] = "1"
    env['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"
    argument = ["--checkpoint=resnet18_finetune", f"--slide_name={slide_name}"]
    print(f"Running inference: python -m inference {' '.join(argument)}")
    result = subprocess.run(
                ["python", "-m", "inference"] + argument, 
                env=env, capture_output=True, text=True) 
    print(f"\nRESULT:\n{result.stdout}ERROR:\n{result.stderr}\n")
    return result



# Custom error handler for 404 errors
@app.errorhandler(404)
def handle_404_error(error):
    response = jsonify({"error": error.description})
    response.status_code = 404
    return response


# Custom error handler for 400 errors
@app.errorhandler(500)
def handle_400_error(error):
    response = jsonify({"error": error.description})
    response.status_code = 500
    return response

def crop_122_train_wsi(d_col=5, d_row=3):
    # Load the image
    image = Image.open('app/images/122_HER2_train_1_2_raw.png')
    image_array = np.array(image)
    img_size = image_array.shape

    # Define the size to divide the image into
    cols, rows = SLIDES_SIZE[2]
    tile_size = (img_size[0]/rows, img_size[1]/cols)

    # Crop the first 3 rows and 5 columns
    cropped_image_array = image_array[int(d_row*tile_size[0]):, int(d_col*tile_size[1]):]
    SLIDES_SIZE[2] = [SLIDES_SIZE[2][0] - d_row, SLIDES_SIZE[2][1] - d_col]

    # Convert back to an image
    cropped_image = Image.fromarray(cropped_image_array)

    # Save the cropped image
    cropped_image.save('app/images/wsi/122_HER2_train_1_2.png')
    return


if __name__ == '__main__':
    # prepare images
    # util.geojson_to_png()
    # util.crop_heatmap()
    crop_122_train_wsi(15, 4)

    app.run(debug=True, host='localhost', port=2000)