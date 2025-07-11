from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import matlab.engine
import numpy as np
from PIL import Image
import io
import base64
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import os
import requests

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Enable CORS for Chrome extension
eng = matlab.engine.start_matlab()

def matlab_array_to_npimg(matlab_array):
    arr = np.array(matlab_array)
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return arr

def matlab_array_to_base64(matlab_array):
    arr = matlab_array_to_npimg(matlab_array)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    byte_data = buf.getvalue()
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    return base64_str

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/process', methods=['POST'])
def process():
    # Accept both files
    original_file = request.files['original']
    filtered_file = request.files.get('filtered')
    filtered_url = request.form.get('filtered_url')

    # Save uploaded files
    original_path = 'uploads/original.png'
    filtered_path = 'uploads/filtered.png'
    os.makedirs('uploads', exist_ok=True)
    original_file.save(original_path)

    if filtered_file:
        filtered_file.save(filtered_path)
    elif filtered_url:
        # Download image with headers to avoid blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        r = requests.get(filtered_url, headers=headers)
        with open(filtered_path, 'wb') as f:
            f.write(r.content)
    else:
        return jsonify({'error': 'No filtered image'}), 400

    # Read images as numpy arrays (for SSIM later)
    original_img = np.array(Image.open(original_path).convert("RGB"))

    # Call MATLAB function
    restored_img = eng.reverse_filter(original_path, filtered_path, nargout=1)

    # Convert MATLAB output to numpy image
    restored_np = matlab_array_to_npimg(restored_img)

    # Ensure size match for SSIM
    if restored_np.shape != original_img.shape:
        restored_np = np.array(Image.fromarray(restored_np).resize((original_img.shape[1], original_img.shape[0])))

    # Convert to grayscale for SSIM
    original_gray = np.array(Image.fromarray(original_img).convert("L"))
    restored_gray = np.array(Image.fromarray(restored_np).convert("L"))

    # Calculate SSIM
    similarity, _ = ssim(original_gray, restored_gray, full=True, win_size=11, data_range=255)
    print(f"Developer Log: SSIM between original and restored = {similarity}")
    similarity_percent = int(similarity * 100)

    # Calculate PSNR
    psnr_value = psnr(original_gray, restored_gray, data_range=255)
    print(f"Developer Log: PSNR between original and restored = {psnr_value:.2f} dB")

    # Convert all images to base64 for frontend display
    def imgfile_to_b64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    original_b64 = imgfile_to_b64(original_path)
    filtered_b64 = imgfile_to_b64(filtered_path)
    restored_b64 = matlab_array_to_base64(restored_img)

    return jsonify({
        "original": f"data:image/png;base64,{original_b64}",
        "filtered": f"data:image/png;base64,{filtered_b64}",
        "restored": f"data:image/png;base64,{restored_b64}",
        "similarity": similarity_percent
    })

if __name__ == '__main__':
    app.run(debug=True)