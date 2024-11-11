import base64
import glob
import math
from flask import Flask, abort, jsonify, render_template, request
import subprocess
import os
from app import utility as util
from utils.path import get_project_root
import numpy as np
from PIL import Image

app = Flask(__name__)
REPRESENTATIVE_SLIDES = {0 : "92_HER2_train", 1 : "0_HER2_train", 2 : "122_HER2_train", 3 : "103_HER2_train"}
SLIDES_SIZE = {0: [46, 24], 1:[52, 30], 2: [52, 20], 3:[46, 25]}

@app.route('/')
def index():
    path = os.path.join('index.html')
    return render_template(path)

@app.route('/get_wsi', methods=['POST'])
def get_WSI():
    '''Get the WSI image and colormap for the given class'''
    num_class = request.get_json().get('class')
    slide_name = REPRESENTATIVE_SLIDES[num_class]

    WSI_path = util.wsi_path(slide_name)
    colormap_path = util.wsi_colormap_path(slide_name)
    
    # Run inference if the image does not exist
    if len(glob.glob(WSI_path)) != 1:
        result = run_inference(slide_name)
        if result.returncode != 0:
            abort(500, description=f"Error running inference: {result.stderr}")
            return result.stderr, 500
    WSI_path = glob.glob(WSI_path)[0]

    if len(glob.glob(colormap_path)) != 1:
        print(f"DRAWING COLORMAP: {colormap_path}")
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

    # 26, 10 / 41, 3 => 28, 12 / 47, 4
    if slide_name == '122_HER2_train':
        xa, xb = 1.2667, -4.9333
        ya, yb = 1.142, 0.57
        pos[0] = round(pos[0]*xa + xb)
        pos[1] = round(pos[1]*ya + yb)
    print(f"pos: col {pos[0]} , row {pos[1]} of ps {SLIDES_SIZE[num_class]}")
    patch_dict = {
        "slide": slide_name,
        "pos": pos,
        "patch": None,
        "attention": None,
        "label": None,
    }
    patch_file = util.patch_path(slide_name, pos)
    if patch_file is None or not os.path.exists(patch_file):
        abort(404, description=f"Patch {pos} not found.")
    
    with open(patch_file, "rb") as image_file:
            patch_dict['patch'] = base64.b64encode(image_file.read()).decode('utf-8')
        
    attention = util.attention_map_path(slide_name, pos)
    if attention is not None:
        # data payload
        with open(attention, "rb") as image_file:
            patch_dict['attention'] = base64.b64encode(image_file.read()).decode('utf-8')
        patch_dict['label'] = attention.split('_')[-1].split('.')[0]
    else:
        print(f"Attention map {pos} not found.")
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

def crop_122_train_wsi(up, right, down, left):
    # Load the image
    image = Image.open('app/images/122_HER2_train_1_2_raw.png')
    image_array = np.array(image)
    img_size = image_array.shape

    # Define the size to divide the image into
    cols, rows = SLIDES_SIZE[2]
    tile_size = (img_size[0]/rows, img_size[1]/cols)

    # Crop the first 3 rows and 5 columns
    up_t, right_t, down_t, left_t = int(up*tile_size[0]), int(right*tile_size[1]), int(down*tile_size[0]), int(left*tile_size[1])
    up_pad, right_pad, down_pad, left_pad = max(0, -up_t), max(0, -right_t), max(0, -down_t), max(0, -left_t)
    print("Padding: ", up_pad, right_pad, down_pad, left_pad)
    padded_image_array = np.pad(image_array, ((up_pad, down_pad), (left_pad, right_pad), (0, 0)), mode='constant', constant_values=150)

    up_crop, right_crop, down_crop, left_crop = max(0, up_t), max(0, right_t), max(0, down_t), max(0, left_t)
    print("Cropping: ", up_crop, right_crop, down_crop, left_crop)
    down_crop = padded_image_array.shape[0] - down_crop
    right_crop = padded_image_array.shape[1] - right_crop
    cropped_image_array = padded_image_array[up_crop:down_crop, left_crop:right_crop]
    print("Cropped image shape: ", cropped_image_array.shape, " from ", image_array.shape)
    SLIDES_SIZE[2] = [SLIDES_SIZE[2][0] - int(up+down), SLIDES_SIZE[2][1] - int(right+left)]
    
    # Convert back to an image
    cropped_image = Image.fromarray(cropped_image_array)

    # Save the cropped image
    cropped_image.save('app/images/wsi/122_HER2_train_1_2.png')
    return


if __name__ == '__main__':
    # prepare images
    # util.geojson_to_png()
    # util.crop_heatmap()
    # crop_122_train_wsi(1, 1, 3, 4.5)
    
    app.run(debug=True, host='localhost', port=2000)