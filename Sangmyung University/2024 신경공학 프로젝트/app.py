import os
import pygame
import hashlib
import threading
import subprocess
import pandas as pd

from PIL import Image
from erp_combination import erp_combination
from erp_combination2 import erp_combination2
# from src.task2 import combination_task2
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Screen resolution fetching and pygame initialization should be done in the main thread
def get_screen_resolution():
    try:
        pygame.init()
        info = pygame.display.Info()
        resolution = (info.current_w, info.current_h)
        pygame.quit()
        return resolution
    except Exception as e:
        print(f"Error getting screen resolution: {e}")
        return 1920, 1080  # Fallback to default resolution

screen_width, screen_height = get_screen_resolution()

# Define the function to ensure the directory exists
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# Set the upload folder path
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')

# Ensure the upload directory exists
ensure_directory_exists(app.config['UPLOAD_FOLDER'])

# Add a custom filter for zip
@app.template_filter('zip')
def zip_filter(a, b, c):
    return zip(a, b, c)

@app.route('/')
def start_page():
    return render_template('start_page.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/step1')
def step1():
    return render_template('step1.html')

@app.route('/step2')
def step2():
    return render_template('step2.html')

@app.route('/step3')
def step3():
    return render_template('step3.html')

# @app.route('/report')
# def report(): 
#     # Dictionary to store images for each best folder (중복 허용)
#     all_images = {
#         f'best_{i}': {'top': [], 'bottom': [], 'combination': []}
#         for i in range(1, 6)
#     }

#     # Dictionary to store image hashes to prevent duplicates
#     image_hashes = {'top': {}, 'bottom': {}}
#     unique_images = {'top': [], 'bottom': []}

#     def get_image_hash(image_path):
#         try:
#             full_path = os.path.join('static', image_path)
#             with Image.open(full_path) as img:
#                 # Convert image to grayscale and resize to small size for comparison
#                 img = img.convert('L').resize((8, 8), Image.Resampling.LANCZOS)
#                 pixels = list(img.getdata())
#                 avg = sum(pixels) / len(pixels)
#                 # Create binary hash
#                 bits = ''.join(['1' if pixel > avg else '0' for pixel in pixels])
#                 return hashlib.md5(bits.encode()).hexdigest()
#         except Exception as e:
#             print(f"Error processing image {image_path}: {e}")
#             return None

#     # List to store unique images for selected_clothes
#     selected_clothes = []
#     base_folder = os.path.join('static', 'images', 'result', 'combination')

#     # Collect all images for best combinations (중복 허용)
#     for i in range(1, 6):
#         best_folder = os.path.join(base_folder, f'best_{i}')
        
#         if os.path.exists(best_folder):
#             for filename in sorted(os.listdir(best_folder)):
#                 relative_path = f'images/result/combination/best_{i}/{filename}'
                
#                 if filename.startswith('T') and filename.endswith('.jpg'):
#                     all_images[f'best_{i}']['top'].append(relative_path)
#                     # Check for duplicates using image hash
#                     img_hash = get_image_hash(relative_path)
#                     if img_hash and img_hash not in image_hashes['top']:
#                         image_hashes['top'][img_hash] = relative_path
#                         unique_images['top'].append(relative_path)
                        
#                 elif filename.startswith('B') and filename.endswith('.jpg'):
#                     all_images[f'best_{i}']['bottom'].append(relative_path)
#                     # Check for duplicates using image hash
#                     img_hash = get_image_hash(relative_path)
#                     if img_hash and img_hash not in image_hashes['bottom']:
#                         image_hashes['bottom'][img_hash] = relative_path
#                         unique_images['bottom'].append(relative_path)
                        
#                 elif filename.startswith('bottom_') and filename.endswith('.jpg'):
#                     all_images[f'best_{i}']['combination'].append(relative_path)

#     # 선택된 옷 이미지 정보 생성 (실제 중복 제거된 이미지만 포함)
#     selected_clothes = []
    
#     # 중복 제거된 상의 이미지 추가
#     for top_img in unique_images['top']:
#         selected_clothes.append({
#             'path': top_img,
#             'description': 'Top'
#         })
    
#     # 중복 제거된 하의 이미지 추가
#     for bottom_img in unique_images['bottom']:
#         selected_clothes.append({
#             'path': bottom_img,
#             'description': 'Bottom'
#         })

#     # HTML 렌더링
#     return render_template('report.html', all_images=all_images, selected_clothes=selected_clothes)

def run_erp_combination_async(*args, **kwargs):
    result = erp_combination(*args, **kwargs)
    if result and len(result) == 2:
        _, recommended_images = result
        session['recommended_images'] = recommended_images

def run_erp_combination2_async(*args, **kwargs):
    try:
        result = erp_combination2(**kwargs)
        if result and len(result) == 2:
            _, recommended_combinations = result
            session['recommended_combinations'] = recommended_combinations
    except Exception as e:
        print(f"Error in ERP combination 2: {e}")
        session['error'] = str(e)

@app.route('/run_erp_combination_top', methods=['POST'])
def run_erp_combination_top():
    run_erp_combination_async(
        screen_width=screen_width,
        screen_height=screen_height,
        fs=256,
        channels=['EEG_Fp1', 'EEG_Fp2'],
        isi=1000,
        background_path='./images/backgrounds/B0.jpg',
        clothes_type='tops',
        image_folder='./images/tops',
        num_trials=1,
        num_images=30,
        event_save_path='./event',
        result_dir='./plot/tops/0',
        lowcut=1.0,
        highcut=30.0,
        tmin=-0.2,
        tmax=1.0,
        mode='all'
    )
    return redirect(url_for('step1'))

@app.route('/run_erp_combination_bottom', methods=['POST'])
def run_erp_combination_bottom():
    run_erp_combination_async(
        screen_width=screen_width,
        screen_height=screen_height,
        fs=256,
        channels=['EEG_Fp1', 'EEG_Fp2'],
        isi=1000,
        background_path='./images/backgrounds/B0.jpg',
        clothes_type='bottoms',
        image_folder='./images/bottoms',
        num_trials=1,
        num_images=30,
        event_save_path='./event',
        result_dir='./plot/bottoms/0',
        lowcut=1.0,
        highcut=30.0,
        tmin=-0.2,
        tmax=1.0,
        mode='all'
    )
    return redirect(url_for('step1'))

@app.route('/run_erp_combination', methods=['POST'])
def run_erp_combination():
    run_erp_combination_async(
        screen_width=screen_width,
        screen_height=screen_height,
        fs=256,
        channels=['EEG_Fp1', 'EEG_Fp2'],
        isi=1000,
        top_image_path='./result/bottoms',
        clothes_type='tops',
        image_folder='./result/tops',
        background_path='./images/backgrounds/B0.jpg',
        num_trials=1,
        num_images=30,
        event_save_path='./event',
        result_dir='./plot/tops/0',
        lowcut=1.0,
        highcut=30.0,
        tmin=-0.2,
        tmax=1.0,
        mode='all'
    )
    return redirect(url_for('step2'))

@app.route('/run_combination_display', methods=['POST'])
def run_combination_display():
    run_erp_combination2_async(
        screen_width=screen_width,
        screen_height=screen_height,
        fs=256,
        channels=['EEG_Fp1', 'EEG_Fp2'],
        isi=1000,
        num_trials=1,
        num_images=25,
        image_folder='./images/chosen_combinations',
        background_path='./images/backgrounds/B0.jpg',
        event_save_path='./event',
        result_dir='./plot/combination/0',
        lowcut=1.0,
        highcut=30.0,
        tmin=-0.2,
        tmax=1.0,
        mode='all'
    )
    return redirect(url_for('step3'))
@app.route('/dress_up')
def dress_up():
    return render_template('dress_up.html')

@app.route('/report')
def report():
    try:
        # Initialize empty dictionary for all_images
        all_images = {
            'best_1': {'tops': [], 'bottoms': []},
            'best_2': {'tops': [], 'bottoms': []},
            'best_3': {'tops': [], 'bottoms': []}
        }

        # Get image filenames for each combination
        folder_path_1 = 'static/images/result/combination/best_1/'
        folder_path_2 = 'static/images/result/combination/best_2/'
        folder_path_3 = 'static/images/result/combination/best_3/'

        # Initialize image variables with None
        top_image_1 = None
        bottom_image_1 = None
        top_image_2 = None
        bottom_image_2 = None
        top_image_3 = None
        bottom_image_3 = None

        selected_images = []

        # Try to get image filenames if directories exist
        try:
            if os.path.exists(folder_path_1):
                top_image_1 = next((f for f in os.listdir(folder_path_1) if f.startswith('T')), None)
                bottom_image_1 = next((f for f in os.listdir(folder_path_1) if f.startswith('B')), None)
                if top_image_1:
                    selected_images.append(f'images/result/combination/best_1/{top_image_1}')
                if bottom_image_1:
                    selected_images.append(f'images/result/combination/best_1/{bottom_image_1}')
            if os.path.exists(folder_path_2):
                top_image_2 = next((f for f in os.listdir(folder_path_2) if f.startswith('T')), None)
                bottom_image_2 = next((f for f in os.listdir(folder_path_2) if f.startswith('B')), None)
                if top_image_2:
                    selected_images.append(f'images/result/combination/best_2/{top_image_2}')
                if bottom_image_2:
                    selected_images.append(f'images/result/combination/best_2/{bottom_image_2}')
            if os.path.exists(folder_path_3):
                top_image_3 = next((f for f in os.listdir(folder_path_3) if f.startswith('T')), None)
                bottom_image_3 = next((f for f in os.listdir(folder_path_3) if f.startswith('B')), None)
                if top_image_3:
                    selected_images.append(f'images/result/combination/best_3/{top_image_3}')
                if bottom_image_3:
                    selected_images.append(f'images/result/combination/best_3/{bottom_image_3}')
        except Exception as e:
            print(f"Error reading image files: {e}")

        # Try to read CSV file
        try:
            df = pd.read_csv('static/data/clothes_info.csv')
            
            for best_num in ['best_1', 'best_2', 'best_3']:
                best_df = df[df['best_num'] == int(best_num[-1])]
                
                # Get actual image files in the directory
                real_image_dir = os.path.join('static', 'images', 'result', 'combination', best_num)
                existing_files = os.listdir(real_image_dir) if os.path.exists(real_image_dir) else []
                
                # Process tops
                tops_df = best_df[best_df['type'] == 'top']
                for _, row in tops_df.iterrows():
                    csv_filename = os.path.basename(row['image_path'])
                    if csv_filename in existing_files:
                        all_images[best_num]['tops'].append({
                            'name': row['name'],
                            'price': row['price'],
                            'color': row['color']
                        })
                
                # Process bottoms
                bottoms_df = best_df[best_df['type'] == 'bottom']
                for _, row in bottoms_df.iterrows():
                    csv_filename = os.path.basename(row['image_path'])
                    if csv_filename in existing_files:
                        all_images[best_num]['bottoms'].append({
                            'name': row['name'],
                            'price': row['price'],
                            'color': row['color']
                        })
                        
        except Exception as e:
            print(f"Error reading CSV file: {e}")

        top_images = [f for f in os.listdir('static/images/result/tops') if os.path.isfile(os.path.join('static/images/result/tops', f))]
        bottom_images = [f for f in os.listdir('static/images/result/bottoms') if os.path.isfile(os.path.join('static/images/result/bottoms', f))]
        
        return render_template('report.html', 
                             all_images=all_images, 
                             top_images=top_images,
                             bottom_images=bottom_images,
                             selected_images=selected_images,
                             top_image_1=top_image_1,
                             bottom_image_1=bottom_image_1,
                             top_image_2=top_image_2,
                             bottom_image_2=bottom_image_2,
                             top_image_3=top_image_3,
                             bottom_image_3=bottom_image_3)
                             
    except Exception as e:
        print(f"Error in report route: {e}")
        empty_images = {
            'best_1': {'tops': [], 'bottoms': []},
            'best_2': {'tops': [], 'bottoms': []},
            'best_3': {'tops': [], 'bottoms': []}
        }
        return render_template('report.html', 
                             all_images=empty_images,
                             top_image_1=None,
                             selected_images=selected_images,
                             bottom_image_1=None,
                             top_image_2=None,
                             bottom_image_2=None,
                             top_image_3=None,
                             bottom_image_3=None,
                             error=str(e))

@app.route('/upload_and_execute_dress_up', methods=['POST'])
def upload_and_execute_dress_up():
    # Clear all files in the upload folder
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            return jsonify(success=False, message=f"Error deleting file: {e}")

    # Check for the model image
    if 'model_image' not in request.files:
        return jsonify(success=False, message="Model image not provided")

    model_image = request.files['model_image']

    if model_image.filename == '':
        return jsonify(success=False, message="No selected file for model image")

    # Save the uploaded image to a temporary location
    model_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'model_img.jpg')
    model_image.save(model_image_path)

    # Execute dress_up.py
    try:
        # Ensure that the subprocess runs until all images are processed and saved
        subprocess.run(['python', 'dress_up.py'], check=True)
        
        # Check if images are saved in the expected directories
        if os.path.exists('static/images/result/combination/best_1') and \
           os.path.exists('static/images/result/combination/best_2') and \
           os.path.exists('static/images/result/combination/best_3'):
            return jsonify(success=True, message="File uploaded and dress up process completed successfully")
        else:
            return jsonify(success=False, message="Image saving failed.")
    except subprocess.CalledProcessError as e:
        return jsonify(success=False, message=f"Error during dress up process: {e}")
# @app.route('/upload_and_execute_dress_up', methods=['POST'])
# def upload_and_execute_dress_up():
#     # Clear all files in the upload folder
#     for filename in os.listdir(app.config['UPLOAD_FOLDER']):
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         try:
#             if os.path.isfile(file_path):
#                 os.unlink(file_path)
#         except Exception as e:
#             return jsonify(success=False, message=f"Error deleting file: {e}")

#     # Check for the model image
#     if 'model_image' not in request.files:
#         return jsonify(success=False, message="Model image not provided")

#     model_image = request.files['model_image']

#     if model_image.filename == '':
#         return jsonify(success=False, message="No selected file for model image")

#     # Save the uploaded image to a temporary location
#     model_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'model_img.jpg')
#     model_image.save(model_image_path)

#     # Execute dress_up.py
#     try:
#         # Ensure that the subprocess runs until all images are processed and saved
#         subprocess.run(['python', 'dress_up.py'], check=True)
        
#         # Check if images are saved in the expected directories
#         # This is a placeholder check; adjust paths as necessary
#         if os.path.exists('static/images/result/combination/best_1') and \
#            os.path.exists('static/images/result/combination/best_2') and \
#            os.path.exists('static/images/result/combination/best_3'):
#             return jsonify(success=True, message="File uploaded and dress up process completed successfully")
#         else:
#             return jsonify(success=False, message="Image saving failed.")
#     except subprocess.CalledProcessError as e:
#         return jsonify(success=False, message=f"Error during dress up process: {e}")

if __name__ == '__main__':
    app.run(debug=True)