import base64

from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from io import BytesIO
from model import MMSegModel, get_clip,KoGPT
import torch
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',type=int,help='port to access from clip server.')
    args = parser.parse_args()
    return args

app = Flask (__name__)
CORS(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seg_model = MMSegModel(
    cfg='./assets/segmentation/swinT_segformer.py',
    checkpoint='./assets/segmentation/weights.pth',
    device=device)


@app.route('/image', methods = ['POST'])
def image():
    data = request.get_json()
    img = Image.open(BytesIO(base64.b64decode(data['image'].split(',')[-1]))).convert('RGB')
    img = np.array(img)

    _,masked_img = seg_model(img)
    
    img_buffered = BytesIO()
    mask_buffered = BytesIO()

    if masked_img is not None:
        masked_img = masked_img.astype(np.uint8)
        masked_img = Image.fromarray(masked_img)
        masked_img.save(mask_buffered,format='JPEG')
        masked_img_str = base64.b64encode(mask_buffered.getvalue())
    else:
        masked_img_str=None

    img = Image.fromarray(img)
    img.save(img_buffered, format="JPEG")
    img_str = base64.b64encode(img_buffered.getvalue())
    
    return {'image':str(img_str),'masked_image':str(masked_img_str)}


if __name__ == "__main__":
    args = parse_args()
    app.run(host='0.0.0.0',port=args.port)

