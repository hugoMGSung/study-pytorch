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
	> pip install pytorch torchvision torchaudio cpuonly
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

[파이토치01](https://github.com/hugoMGSung/study-pytorch/blob/main/pytorch01/Pytorch01.md)