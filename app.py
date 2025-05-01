from flask import Flask, request, send_file, render_template, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np
from scipy import fftpack
import io
import os

app = Flask(__name__, static_folder='.', template_folder='.')  # Serve HTML/CSS/JS
CORS(app)

def reverse_filter(image: Image.Image, iterations=20, lambda_val=0.01, kernel_size=5, sigma=1.5, return_buffer=True):
    """
    Apply blind deconvolution to restore an image degraded by an unknown filter.
    
    Args:
        image: PIL Image object of the filtered/degraded image
        iterations: Number of iterations for the restoration algorithm
        lambda_val: Regularization parameter to stabilize deconvolution
        kernel_size: Size of the estimated blur kernel
        sigma: Standard deviation for the Gaussian blur kernel estimation
        return_buffer: If True, returns BytesIO buffer; if False, returns PIL Image
    
    Returns:
        BytesIO buffer containing the restored image (if return_buffer=True)
        PIL Image object of the restored image (if return_buffer=False)
    """
    # Convert PIL image to numpy array and separate channels
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Handle grayscale images
    if len(img_array.shape) == 2:
        is_grayscale = True
        # Add channel dimension for consistent processing
        img_array = np.expand_dims(img_array, axis=2)
    else:
        is_grayscale = False
    
    height, width, channels = img_array.shape
    
    # Create initial estimate of the blur kernel (PSF)
    x, y = np.meshgrid(np.arange(-kernel_size//2 + 1, kernel_size//2 + 1), 
                       np.arange(-kernel_size//2 + 1, kernel_size//2 + 1))
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)  # Normalize
    
    # Pad kernel to image size for FFT
    padded_kernel = np.zeros((height, width))
    kh, kw = kernel.shape
    padded_kernel[:kh, :kw] = kernel
    padded_kernel = np.roll(padded_kernel, (-kh//2, -kw//2), axis=(0, 1))
    
    # Process each channel
    restored_channels = []
    for c in range(channels):
        channel = img_array[:, :, c]
        
        # Initialize with the filtered image
        x_cur = channel.copy()
        
        # FFT of kernel
        H = fftpack.fft2(padded_kernel)
        
        # Iterative restoration
        for _ in range(iterations):
            # Apply estimated degradation to current estimate
            x_f_cur = np.real(fftpack.ifft2(fftpack.fft2(x_cur) * H))
            
            # First-order reverse filtering with regularization
            numerator = fftpack.fft2(channel) * np.conj(H)
            denominator = np.abs(H)**2 + lambda_val
            x_cur = np.real(fftpack.ifft2(numerator / denominator))
            
            # Ensure values are in valid range
            x_cur = np.clip(x_cur, 0, 1)
        
        restored_channels.append(x_cur)
    
    # Combine channels
    if is_grayscale:
        restored_img = restored_channels[0]
    else:
        restored_img = np.stack(restored_channels, axis=2)
    
    # Convert back to 8-bit and then to PIL Image
    restored_img = (restored_img * 255).astype(np.uint8)
    if is_grayscale:
        restored_pil = Image.fromarray(restored_img)
    else:
        restored_pil = Image.fromarray(restored_img)
    
    # Return either PIL Image or BytesIO buffer
    if return_buffer:
        buf = io.BytesIO()
        restored_pil.save(buf, format='PNG')
        buf.seek(0)
        return buf
    else:
        return restored_pil

# @app.route('/api/reverse-filter', methods=['POST'])
# def reverse_filter_endpoint():
#     if 'image' not in request.files:
#         return "No image uploaded", 400

#     try:
#         img = Image.open(request.files['image'])
#         buf = reverse_filter(img)
#         return send_file(buf, mimetype='image/png')
#     except Exception as e:
#         import traceback
#         print(f"Error processing image: {str(e)}")
#         print(traceback.format_exc())  # Print detailed error trace
#         return f"Error processing image: {str(e)}", 500

# # Add a simple route for testing
# @app.route('/')
# def home():
#     return "Reverse Filter API is running!"

# if __name__ == '__main__':
#     print("Starting Flask server on port 8000...")
#     app.run(debug=True, host='0.0.0.0', port=8000)  # Added host parameter for better accessibility

# if __name__ == '__main__':
#     app.run(debug=True, port=8000)  # Changed port to 8000 to match your server

@app.route('/')
def index():
    return send_file('main.html')  # Serve your HTML page

@app.route('/styles.css')
def styles():
    return send_file('styles.css')

@app.route('/images/<path:filename>')
def serve_images(filename):
    return send_from_directory('images', filename)

@app.route('/api/reverse-filter', methods=['POST'])
def reverse_filter_endpoint():
    if 'image' not in request.files:
        return "No image uploaded", 400
    try:
        img = Image.open(request.files['image'])
        buf = reverse_filter(img)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return f"Error processing image: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)