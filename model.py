import torch
import cv2
import numpy as np
import clip
import re

from transformers import PreTrainedTokenizerFast,GPT2LMHeadModel
class KoGPT:
    def __init__(self,pretrained='larcane/kogpt2-cat-diary',device='cpu'):
        self.model = GPT2LMHeadModel.from_pretrained(pretrained).to(device)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained,
                    bos_token='<s>', eos_token='</s>', unk_token='<unk>',
                    pad_token='<pad>', mask_token='<mask>')
        self.device=device

    def __call__(self,label,max_length=128,top_k=200,top_p=0.95):
        token,head = self.preprocess(label)
        input_ids = self.tokenizer.encode(token,return_tensors='pt').to(self.device)
        generated = self.model.generate(
            input_ids,
            do_sample=True,
            num_return_sequences=1,
            max_length=max_length, 
            top_k=top_k, 
            top_p=top_p,
            # temperature=0.9,
            # eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,
            bad_words_ids=[[self.tokenizer.unk_token_id]]
        ).tolist()
        text = self.tokenizer.decode(generated[0],skip_special_tokens=True)
        return self.postprocess(text,head)

    def preprocess(self,label):
        head = label.strip()+"의 일기 ::"
        return f"<s>{head}</s><s>",head

    def postprocess(self,text,head):
        return re.sub(head,'',text).strip()



import mmcv
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor
from mmcv.parallel import MMDataParallel
from mmseg.datasets.pipelines import Compose
import cv2

class MMSegModel:
    def __init__(self,cfg,checkpoint,device):
        cfg = mmcv.Config.fromfile(cfg)
        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        cfg.data.test.test_mode = True

        segmentor = build_segmentor(
            cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
        # segmentor.load_state_dict(a['state_dict'])
        load_checkpoint(segmentor,checkpoint, map_location='cpu')
        segmentor.eval()
        self.segmentor = MMDataParallel(segmentor, device_ids=[0]).to(device)
        self.transform = Compose([cfg.data.test['pipeline'][1]])
        self.device=device

    def __call__(self,image):
        masked_image = image.copy()
        mask = self.segmenting(image)  # x shape : H,W,3
        try:
            x1 = np.where(mask.max(axis=0)==1)[0].min()
        except: x1=0
        try:
            y1 = np.where(mask.max(axis=1)==1)[0].min()
        except: y1=0
        try:
            x2 = np.where(mask.max(axis=0)==1)[0].max()
        except: x2=mask.shape[0]
        try:
            y2 = np.where(mask.max(axis=1)==1)[0].max()
        except: y2=mask.shape[1]

        if mask.sum().item() < 30:
            return (None,None)
        masked_image[mask==0]=255 # HxWx3
        masked_image = masked_image[y1:y2,x1:x2,:]
        return (image,masked_image)

        
    def segmenting(self,x):
        data = self.preprocess(x)
        data = self.transform(data)
        data = self.postprocess(data)
        return self.segmentor.forward(return_loss=False,**data)[0]

    def preprocess(self,data):
        return dict(img=data,filename='',ori_filename='',ori_shape=data.shape)

    def postprocess(self,data):
        data['img_metas'][0]._data=[[data['img_metas'][0]._data]]
        data['img'][0] = data['img'][0].unsqueeze(0)
        return data



def get_clip(device):
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    return clip_model,preprocess
