from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import albumentations as A
import albumentations.pytorch as AP
import base64
import numpy as np
import cv2
import matplotlib.pyplot as plt
import clip
import torch
from model import MMSegModel, get_clip,KoGPT
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=6006,help='port number to access from middleware or front')
    parser.add_argument('--proxy',default=False,action='store_true',help='If proxy is True, not parsing data to json format; otherwise parsing data to json format.')
    args = parser.parse_args()
    return args


app = Flask (__name__)
CORS(app)

## model load
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seg_model = MMSegModel(
    cfg='./assets/segmentation/swinT_segformer.py',
    checkpoint='./assets/segmentation/weights.pth',
    device=device)

gpt_model = KoGPT(pretrained='larcane/kogpt2-cat-diary',device=device)

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
    'a picture of sitting cat',
    'a picture of standing cat',
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
    'a picture of standing cat':'서있는 고양이',
    'a picture of sitting cat':'앉아있는 고양이',
}


action_tokens = clip.tokenize(actions).to(device)
emotion_tokens = clip.tokenize(emotions).to(device)
cat_token = clip.tokenize(['cat']).to(device)

@app.route('/hello')
def hello():
    return jsonify({
        'image':"",
        'text':'hello',
    })

@app.route('/image', methods = ['POST'])
def image():
    data = request.get_json()
    img = Image.open(BytesIO(base64.b64decode(data['image'].split(',')[-1]))).convert('RGB')
    img = np.array(img)


    _,masked_img = seg_model(img)
    if masked_img is None:
        text = f"고양이가 아니다냥!\nfilter in segmentation model"
    else:
        masked_img = masked_img.astype(np.uint8)

        if isinstance(preprocess,A.Compose):
            processed_img = preprocess(image=img)['image'].unsqueeze(0).to(device)
            processed_masked_img = preprocess(image=masked_img)['image'].unsqueeze(0).to(device)
        else:
            processed_img = preprocess(img).unsqueeze(0).to(device)
            processed_masked_img = preprocess(masked_img).unsqueeze(0).to(device)

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

                action_text = gpt_model(kor_action)
                emotion_text = gpt_model(kor_emotion)
                text=f"action:[{kor_action}]{action_text}\n\nemotion:[{kor_emotion}]{emotion_text}"
            else:
                text=f"고양이가 아니다냥!\ncat similarity:[{cat_logit}]"

    if args.proxy:
        return text
    else:
        return {'text':text}

if __name__ == "__main__":
    args = parse_args()
    app.run(host='0.0.0.0',port=args.port)