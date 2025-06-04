from flask import Flask, request, send_file
import os

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    file.save('input.png')  # 存下来准备传给MATLAB

    # 调用MATLAB处理
    os.system('matlab -batch "your_matlab_function(\'input.png\', \'output.png\')"')

    # 处理完，把output.png返回
    return send_file('output.png', mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True)
