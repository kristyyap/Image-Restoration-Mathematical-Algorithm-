from flask import Flask, render_template, request, jsonify
import matlab.engine
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__, static_folder='static', template_folder='templates')
eng = matlab.engine.start_matlab()

def matlab_array_to_base64(matlab_array):
    # Convert MATLAB double (values 0-1) to uint8 image array
    arr = np.array(matlab_array)
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
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

    restored_img, filtered_img = eng.reverse_filter(input_path, filter_type, nargout=2)

    # Convert to base64
    filtered_b64 = matlab_array_to_base64(filtered_img)
    restored_b64 = matlab_array_to_base64(restored_img)

    return jsonify({
        "filtered": f"data:image/png;base64,{filtered_b64}",
        "restored": f"data:image/png;base64,{restored_b64}"
    })

if __name__ == '__main__':
    app.run(debug=True)
