from model import get_clip
from flask import Flask, request
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import base64
import numpy as np
import torch
import albumentations as A
import albumentations.pytorch as AP
import clip
import requests
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg-server',type=str,default='http://localhost:8000/image',help='segmentation server http address.')
    parser.add_argument('--port',type=int,default=8001,help='port number to access from gpt2 server.')

    args = parser.parse_args()
    return args

app = Flask (__name__)
CORS(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clip_model, _ = get_clip(device)
_ = clip_model.eval()

preprocess = A.Compose([
    A.SmallestMaxSize(clip_model.visual.input_resolution),
    A.CenterCrop(clip_model.visual.input_resolution,clip_model.visual.input_resolution),
    A.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    AP.ToTensorV2()
])


actions = [
    'a picture of cat lying down',
    'a picture of cat lying on side',
    'a picture of grooming cat',
    'a picture of playing cat',
    'a picture of cat punching',
    'a picture of cat eating',
    # 'hugged cat',
    # 'box cat',
    # 'liquid cat',
    # 'upside down cat',
]

emotions = [
    'a picture of sleeping cat',
    'a picture of sleepy cat', # 누워서 자고 있지 않은 것 (하품, 앉아서 자는 고양이 등)
    'a picture of happy cat',
    'a picture of comfortable cat',
    'a picture of curious cat',
    'a picture of perplexed cat',
    'a picture of scared cat',
    'a picture of sad cat',
    'a picture of cloudy cat',
    'a picture of angry cat',
]

labels_eng2kor = {
    'a picture of cat lying down':"엎드려있는 고양이",
    'a picture of cat lying on side':"옆으로 누워있는 고양이",
    'a picture of grooming cat' :"그루밍하는 고양이",
    'a picture of playing cat' :"놀고있는 고양이",
    'a picture of cat punching' :"펀치를 하는 고양이",
    'a picture of cat eating' :"밥을 먹는 고양이",
    'a picture of sleeping cat' :"자고 있는 고양이",
    'a picture of sleepy cat':"졸린 고양이", 
    'a picture of happy cat' :"행복한 고양이",
    'a picture of comfortable cat' :"편안한 고양이",
    'a picture of curious cat':"궁금해하는 고양이",
    'a picture of perplexed cat':"당황한 고양이",
    'a picture of scared cat':"무서워하는 고양이",
    'a picture of sad cat':"슬픈 고양이",
    'a picture of cloudy cat':"언짢은 고양이",
    'a picture of angry cat':"화난 고양이",
}

action_tokens = clip.tokenize(actions).to(device)
emotion_tokens = clip.tokenize(emotions).to(device)
cat_token = clip.tokenize(['cat']).to(device)


@app.route('/image', methods = ['POST'])
def image():
    res = requests.post(args.seg_server,json=request.get_json())
    data =res.json()
    img_bytes = BytesIO(base64.b64decode(data['image'][1:]))
    masked_img_bytes = BytesIO(base64.b64decode(data['masked_image'][1:]))

    img = Image.open(img_bytes).convert('RGB')
    img = np.array(img)

    if masked_img_bytes is not None:
        masked_img = Image.open(masked_img_bytes).convert('RGB')
        masked_img = np.array(masked_img)

        processed_img = preprocess(image=img)['image'].unsqueeze(0).to(device)
        processed_masked_img = preprocess(image=masked_img)['image'].unsqueeze(0).to(device)


        # 0. 고양이 사진인지 체크
        # 1. masked_img와 emotion token/ action token
        # 2. emotion token이 큰 순서대로 action token에 곱함
        # 3. img로 action token
        # 4. action token끼리 곱
        # 5. 최대 token 출력

        with torch.no_grad():
            image_features = clip_model.encode_image(torch.cat([processed_masked_img,processed_img],dim=0))
            text_features = clip_model.encode_text(torch.cat([action_tokens,emotion_tokens,cat_token],dim=0))

            # normalized features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            masked_img_logit,origin_img_logit = image_features@text_features.T

            cat_logit = origin_img_logit.squeeze()[-1]
            if cat_logit>=0.2:
                masked_action_logit = masked_img_logit.squeeze()[:len(actions)]
                masked_emotion_logit = masked_img_logit.squeeze()[len(actions):-1]
                origin_action_logit = origin_img_logit.squeeze()[:len(actions)]
                origin_emotion_logit = origin_img_logit.squeeze()[len(actions):-1]

                masked_pred_emotion = emotions[masked_emotion_logit.argmax()]
                masked_pred_action = actions[masked_action_logit.argmax()]
                origin_pred_emotion = emotions[origin_emotion_logit.argmax()]
                origin_pred_action = actions[origin_action_logit.argmax()]
                com_pred_action =  actions[(origin_action_logit*masked_action_logit).argmax()]

                kor_action = labels_eng2kor[com_pred_action]
                kor_emotion = labels_eng2kor[masked_pred_emotion]
            else:
                kor_action = kor_emotion = None
    else:
        kor_action = kor_emotion = None

    buffered = BytesIO()
    img = Image.fromarray(img)
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())

    return {'image': str(img_str), 'kor_action':kor_action,'kor_emotion':kor_emotion}

if __name__ == "__main__":
    args = parse_args()
    app.run(host='0.0.0.0',port=args.port)