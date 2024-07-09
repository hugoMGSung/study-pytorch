# study-pytorch
PyTorch 학습 리포지토리

## 파이토치 시작

### 개요
- 2016년 페이스북이 발표한 오픈소스 기반 딥러닝 프레임워크
	- 구글 Tensorflow와 양대산맥
	- YOLO v8을 파이토치로 개발하여 Object Detection 분야에 급격하게 성장중

	![파이토치아키텍처](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcpePg2%2Fbtr3J211WJJ%2F0fKTbyJShsV6kWcUeFEhlk%2Fimg.png)

### 설치
- 기본설치
	- https://pytorch.org/ 에서 플랫폼 별로 명령어 검색가능

	```shell
	> pip install torch torchvision torchaudio
	```

	- 나의 경우 CUDA 12 버전 설치됨
	```shell
	> nvcc --version
	nvcc: NVIDIA (R) Cuda compiler driver
	Copyright (c) 2005-2023 NVIDIA Corporation
	Built on Tue_Jul_11_03:10:21_Pacific_Daylight_Time_2023
	Cuda compilation tools, release 12.2, V12.2.128
	Build cuda_12.2.r12.2/compiler.33053471_0

	> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 
	```

### 설치 확인
- CUDA 확인
	- 권장버전은 12.4 인데 현재 11.2가 설치되어 있음
	- PyTorch에서는 12.1을 권장. 다시 CUDA 설치함

	<img src="https://raw.githubusercontent.com/hugoMGSung/study-pytorch/main/images/torch0001.png" width="730">


- PyTorch + Cuda
	```shell
	(pytorch_env) PS > pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
	...
	Installing collected packages: torch, torchvision, torchaudio
	Successfully installed torch-2.3.1+cu121 torchaudio-2.3.1+cu121 torchvision-0.18.1+cu121
	(pytorch_env) PS >
	```

	- 추가 - numpy는 1.X 버전으로 재설치 할 것


[파이토치01](https://github.com/hugoMGSung/study-pytorch/blob/main/pytorch01/Pytorch01.ipynb)


## 기본 내용 중

### 판다스...
- 판다스, 넘파이, 맷플롭립은 패스

## 파이토치 패키지 구성

### torch
- 메인 네임스페이스. 텐서 등의 수학함수 포함, Numpy와 유사한 구조

### torch.autograd
- 자동 미분을 위한 함수들 포함. 

### torch.nn
- 신경망을 구축하기 위한 다양한 데이터 구조와 레이어 등 정의. RNN, LSTM 레이어와 ReLU, MESLoss 등의 텐서플로우에 있던 함수 모델포함

### torch.optim
- 확률적 경사 하강법(Stochastic Gradient Descent, SGD) 중심 파라미터 최적화 알고리즘 포함

### torch.utils.data
- SGD 반복 연산시 배치용 유틸리티 함수 포함

### torch.onnx
- ONNX(Open Neural Network Exchange) 포맷의 모델 포함


[파이토치2 - 텐서](https://github.com/hugoMGSung/study-pytorch/blob/main/pytorch01/pytorch02.ipynb)

