from flask import Flask, request, send_file, jsonify
from PIL import Image
import numpy as np
import cv2
import io
from scipy.signal import convolve2d
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def motion_blur_kernel(size=10, angle=45):
    kernel = np.zeros((size, size))
    center = size // 2
    angle_rad = np.deg2rad(angle)
    x = int(center + center * np.cos(angle_rad))
    y = int(center + center * np.sin(angle_rad))
    cv2.line(kernel, (center, center), (x, y), 1, thickness=1)
    kernel /= np.sum(kernel)
    return kernel

def apply_filter(img_channel, kernel):
    return convolve2d(img_channel, kernel, mode='same', boundary='wrap')

def reverse_filtering(observed, filter_func, iterations=20):
    current = observed.copy()
    for _ in range(iterations):
        filtered_current = filter_func(current)
        numerator = np.fft.fft2(observed) * np.fft.fft2(current)
        denominator = np.fft.fft2(filtered_current) + 1e-8
        current = np.real(np.fft.ifft2(numerator / denominator))
    return current

def process_image(image_pil):
    img = np.array(image_pil.convert("RGB")) / 255.0
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]

    kernel = motion_blur_kernel(20, 0)
    f = lambda x: apply_filter(x, kernel)

    Ry, Gy, By = R, G, B 
    Xrec_R = reverse_filtering(Ry, f)
    Xrec_G = reverse_filtering(Gy, f)
    Xrec_B = reverse_filtering(By, f)

    restored = np.stack([Xrec_R, Xrec_G, Xrec_B], axis=-1)
    restored = np.clip(restored * 255, 0, 255).astype(np.uint8)
    restored_pil = Image.fromarray(restored)
    
    buf = io.BytesIO()
    restored_pil.save(buf, format="PNG")
    buf.seek(0)
    return buf

@app.route('/api/reverse-filter', methods=['POST'])
def reverse_filter_api():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = Image.open(request.files['image'])
    buf = process_image(image)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=True, host="0.0.0.0")
