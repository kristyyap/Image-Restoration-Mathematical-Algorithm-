from flask import Flask, render_template, request, jsonify
import matlab.engine
import numpy as np
from PIL import Image
import io
import base64
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

app = Flask(__name__, static_folder='static', template_folder='templates')
eng = matlab.engine.start_matlab()

def matlab_array_to_npimg(matlab_array):
    arr = np.array(matlab_array)
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    if arr.ndim == 2:  # Grayscale
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
    file = request.files['image']
    filter_type = request.form.get('filter', 'motion')
    input_path = 'uploaded_image.png'
    file.save(input_path)

    # Get the original image as numpy array for SSIM calculation
    original_img = np.array(Image.open(input_path).convert("RGB"))

    # Call MATLAB function
    restored_img, filtered_img = eng.reverse_filter(input_path, filter_type, nargout=2)

    # Convert MATLAB output to numpy images
    restored_np = matlab_array_to_npimg(restored_img)

    # Ensure both images are the same size for SSIM
    if restored_np.shape != original_img.shape:
        # Resize restored to match original if necessary
        restored_np = np.array(Image.fromarray(restored_np).resize(
            (original_img.shape[1], original_img.shape[0])
        ))

    # Convert to grayscale for SSIM
    original_gray = np.array(Image.fromarray(original_img).convert("L"))
    restored_gray = np.array(Image.fromarray(restored_np).convert("L"))

    # Calculate SSIM (value in range [-1, 1])
    similarity, _ = ssim(original_gray, restored_gray, full=True)
    similarity_percent = int(similarity * 100)

    # Calculate PSNR (in dB)
    psnr_value = psnr(original_gray, restored_gray, data_range=255)
    print(f"Developer Log: PSNR between original and restored = {psnr_value:.2f} dB")

    # Convert to base64 for frontend
    filtered_b64 = matlab_array_to_base64(filtered_img)
    restored_b64 = matlab_array_to_base64(restored_img)

    return jsonify({
        "filtered": f"data:image/png;base64,{filtered_b64}",
        "restored": f"data:image/png;base64,{restored_b64}",
        "similarity": similarity_percent
    })

if __name__ == '__main__':
    app.run(debug=True)
