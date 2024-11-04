import base64
from flask import Flask, abort, jsonify, render_template, request, send_file
import subprocess
import os
from utils.path import get_project_root, get_labeled_WSI_files

app = Flask(__name__)

@app.route('/')
def index():
    path = os.path.join('index.html')
    return render_template(path)

@app.route('/run_python', methods=['POST'])
def run_python():
    # Run inference code
    script_path = "inference"
    slide_name = request.get_json().get('slide_name')
    available_slides = get_labeled_WSI_files(type='train') + get_labeled_WSI_files(type='val') + get_labeled_WSI_files(type='test')
    available_slides = [slide.split('/')[-1].split('.')[0] for slide in available_slides]
    if slide_name not in available_slides:
        abort(404, description=f"Slide {slide_name} not found.") # Available slides: {available_slides}

    env = os.environ.copy()
    env['PYTHONPATH'] = f"{get_project_root()}"
    env['CUDA_VISIBLE_DEVICES'] = "1"
    env['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"
    argument = ["--checkpoint=resnet18_finetune", f"--slide_name={slide_name}"]
    print(f"Running inference: python -m {script_path} {' '.join(argument)}")
    result = subprocess.run(["python", "-m", script_path] + argument, 
                   env=env, capture_output=True, text=True) 
    print(f"\nRESULT:\n{result.stdout}ERROR:\n{result.stderr}\n")
    
    if result.returncode != 0:
        abort(500, description=f"Error running inference: {result.stderr}")
        return result.stderr, 500
    else:
        output_image_path = f"{get_project_root()}/test/output/{slide_name}.png"
        colormap_image_path = f"{get_project_root()}/test/output/{slide_name}_colormap.png"
        # Create your data payload
        with open(output_image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        with open(colormap_image_path, "rb") as image_file:
            encoded_colormap = base64.b64encode(image_file.read()).decode('utf-8')
        detected_label = result.stdout.split('detected as: ')[1].split(' (True label:')[0]
        true_label = result.stdout.split('(True label: ')[1].split(')')[0]
        message = {"detected": detected_label, 
                   "true": true_label, 
                   "image": encoded_image,
                   "colormap": encoded_colormap
                   }
        return jsonify(message)



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


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=3000)