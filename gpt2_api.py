import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import base64
import numpy as np
import torch
from model import KoGPT
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip-server',type=str,default='http://localhost:8000/image',help='clip server http address.')
    parser.add_argument('--port',type=int,default=6006,help='port number to access from middleware or front')
    parser.add_argument('--proxy',default=False,action='store_true',help='If proxy is True, not parsing data to json format; otherwise parsing data to json format.')
    args = parser.parse_args()
    return args

app = Flask (__name__)
CORS(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gpt_model = KoGPT(pretrained='larcane/kogpt2-cat-diary',device=device)

@app.route('/image', methods = ['POST'])
def image():
    res = requests.post(args.clip_server,json=request.get_json())
    data = res.json()
    kor_action = data['kor_action']
    kor_emotion = data['kor_emotion']
    if kor_action is not None and kor_emotion is not None:
        action_text = gpt_model(kor_action)
        emotion_text = gpt_model(kor_emotion)
        text=f"action:[{kor_action}]{action_text}\n\nemotion:[{kor_emotion}]{emotion_text}"
    else:
        text= f"고양이가 아니다냥!!"
        
    if args.proxy:
        return text
    else:
        return {'text':text}

if __name__ == "__main__":
    args = parse_args()
    app.run(host='0.0.0.0',port=args.port)