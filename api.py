from flask import Flask, request, jsonify
from flask_cors import CORS
from io import BytesIO
from PIL import Image
from io import BytesIO
import base64
import numpy as np
import cv2
import matplotlib.pyplot as plt

app = Flask (__name__)
CORS(app)

@app.route('/hello')
def hello():
    return jsonify({
        'image':"",
        'text':'hello',
    })

@app.route('/image', methods = ['POST'])
def image():
    data = request.get_json()
    img = Image.open(BytesIO(base64.b64decode(data['image'].split(',')[-1])))
    img = img.convert('L')

    plt.imshow(img)

    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())

    return {'image': str(img_str), 'text':'test text'}

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000)