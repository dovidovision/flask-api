import torch
import clip
from torch.autograd.grad_mode import no_grad
from torchvision import transforms as T
from PIL import Image

import os
import math
import time
from .models import TextEncoder
from transformers import AutoTokenizer
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

isHalf = True


class Tester():
    def __init__(self, PATH='/opt/ml/FinalProject/koclip-train/text_encoder_ViT.pth'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('[Model]: loading weights.. ')
        self.ImageEncoder, self.T = clip.load("ViT-B/32", device=self.device)
        self.TextEncoder = TextEncoder(train_projection=False).to(self.device)
        self.Tokenizer = AutoTokenizer.from_pretrained("klue/roberta-small", use_fast=True)

        self.labels = [

            '엎드려있는 고양이',
            '옆으로 누워있는 고양이',
            '앉아있는 고양이', # 앞발은 꼿꼿하고 뒷발은 웅크린 상태
            '서 있는 고양이',
            '안겨있는 고양이',
            '얼굴만 보이는 고양이',
            ' ',

            '박스 고양이',
            '액체 고양이',
            '식빵을 굽는 고양이',
            '엉덩이를 치켜든 고양이',
            '양말을 신은 고양이',
            '무장해제한 고양이',
            '친구와 함께 있는 고양이',
            '그루밍하는 고양이',
            '간식을 먹는 고양이',
            '냥냥펀치를 하는 고양이',
            '놀고 있는 고양이',
            
            '자고 있는 고양이',
            '졸린 고양이', # 누워서 자고 있지 않은 것 (하품, 앉아서 자는 고양이 등)
            '행복한 고양이',
            '편안한 고양이',
            '호기심에 가득 찬 고양이',
            '당황한 고양이',
            '무서워하는 고양이',
            '슬픈 고양이',
            '아무 생각이 없는 고양이',
            '얹짢은 고양이',
            '불안한 고양이',
            '화난 고양이',
        ]
        self.PATH = PATH
        self._get_pretrained()
        if isHalf:
            self._half_precision()
        self.text_embedding = self._get_textEmb()
    
    def _get_pretrained(self):
        startTime = time.time()
        if os.path.isfile(self.PATH):
            self.TextEncoder.load_state_dict(torch.load(self.PATH))
        print(f'[Model]: Done! {int(time.time()-startTime)}sec')
    
    def _half_precision(self):
        self.ImageEncoder = self.ImageEncoder.half()
        self.TextEncoder = TextEncoder(train_projection=False).to(self.device).half()

    
    def _preprocess(self, img):
        n_px = 224
        trans = T.Compose([
            T.Resize((n_px, n_px)),
            lambda img: img.convert('RGB'),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        return trans(img).unsqueeze(0)

    def _get_textEmb(self):
        text_tensor = self.Tokenizer(
            self.labels,
            return_tensors='pt',
            max_length=self.Tokenizer.model_max_length,
            padding="max_length",
            add_special_tokens=True,
            return_token_type_ids=False
        )
        batch_input_ids = text_tensor['input_ids']
        batch_attention_mask = text_tensor['attention_mask']
        if isHalf:
            text_embedding = self.TextEncoder(batch_input_ids.to(self.device), batch_attention_mask.to(self.device)).half()
        else:
            text_embedding = self.TextEncoder(batch_input_ids.to(self.device), batch_attention_mask.to(self.device)).float()
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

        return text_embedding.T

    def get_result(self, img)-> str:
        img = self._preprocess(img)
        with no_grad():
            if isHalf:
                image_embedding = self.ImageEncoder.encode_image(img.to(self.device)) # Output : N x 512
            else:
                image_embedding = self.ImageEncoder.encode_image(img.to(self.device)).float() # Output : N x 512
        
        # Normalization is need for calculating cosine similarity
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)    
        self.text_embedding = self._get_textEmb()
        img2text = (image_embedding @ self.text_embedding) * math.exp(0.07)
        _, pred = torch.max(img2text, 1)

        return self.labels[pred]


if __name__ == '__main__': 
    tester = Tester()
    while True:
        fpath = input('>>')
        startTime = time.time()
        img = Image.open('./00000391_017.jpg')
        print(tester.get_result(img))
        print(time.time() - startTime)