# API
ko-gpt2, ko-clip, segmentation models을 사용하기 위한 api repository입니다.
<br>
<br>
<br>

__api.py__ : 한 서버에서 모든 모델을 구동할 수 있도록 만든 파일(not recommend : anti-arcitecture)
```
--port : server port를 설정할 수 있습니다.
--proxy : proxy server 여부를 설정할 수 있습니다. proxy를 true로 설정한 경우 server에서 json parsing을 하지 않고 데이터를 전송합니다.
```

<br>

__gpt2_api.py__ : gpt2 model API server 구동을 위한 파일
```
--port : server port를 설정할 수 있습니다.
--proxy : proxy server 여부를 설정할 수 있습니다. proxy를 true로 설정한 경우 server에서 json parsing을 하지 않고 데이터를 전송합니다.
--clip-server : clip model의 api address를 입력받습니다. clip_api server ip와 clip_api.py에서 설정한 port와 동일해야 합니다.
```

<br>

__clip_api.py__ : clip model API server 구동을 위한 파일
```
--port : server port를 설정할 수 있습니다.
--proxy : proxy server 여부를 설정할 수 있습니다. proxy를 true로 설정한 경우 server에서 json parsing을 하지 않고 데이터를 전송합니다.
--seg-server : segmentation model의 api address를 입력받습니다. seg_api server ip와 seg_api.py에서 설정한 port와 동일해야 합니다.
```

<br>

__seg_api.py__ : segmentation model API server 구동을 위한 파일
```
--port : server port를 설정할 수 있습니다.
```

<br>
<br>

## Requirements
__api.py__
```
albumentations==1.1.0
clip==1.0
Flask==2.0.2
Flask-Cors==3.0.10
mmcv-full==1.4.1
mmsegmentation==0.20.2
transformers==4.12.5
```

__gpt2_api.py__
```
Flask==2.0.2
Flask-Cors==3.0.10
transformers==4.12.5
```

__clip_api.py__
```
albumentations==1.1.0
clip==1.0
Flask==2.0.2
Flask-Cors==3.0.10
```

__seg_api.py__
```
albumentations==1.1.0
Flask==2.0.2
Flask-Cors==3.0.10
mmcv-full==1.4.1
mmsegmentation==0.20.2
```