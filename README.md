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


#### 가끔씩 실행 오류
- import torch에서 NameError: name '_C' is not defined 가 발생하면
	```shell
	> pip install Cython
	``` 
	
	- 설치 또는 런타임 재실행 하면 됨

## 기본 내용 중

### 판다스...
- 판다스, 넘파이, 맷플롭립은 패스

## 파이토치 패키지 구성

### torch
- 메인 네임스페이스. 텐서 등의 수학함수 포함, Numpy와 유사한 구조

### torch.autograd
- 자동 미분을 위한 함수들 포함. 자동 미분 사용을 위한 컨텍스트 매니저, 기반클래스 Function 등 포함

### torch.nn / torch.nn.functional 
- 신경망을 구축하기 위한 다양한 데이터 구조와 레이어 등 정의. RNN, LSTM 레이어와 ReLU, MESLoss 등의 텐서플로우에 있던 함수 모델포함

### torch.optim
- 확률적 경사 하강법(Stochastic Gradient Descent, SGD) 중심 파라미터 최적화 알고리즘 포함

### torch.utils.data
- SGD 반복 연산시 배치용 유틸리티 함수 포함

### torch.onnx
- ONNX(Open Neural Network Exchange) 포맷의 모델 포함


[파이토치2 - 텐서](https://github.com/hugoMGSung/study-pytorch/blob/main/pytorch01/pytorch02.ipynb)

[선형회귀](https://github.com/hugoMGSung/study-pytorch/blob/main/pytorch01/pytorch03.ipynb)



[인경신공망](https://github.com/hugoMGSung/study-pytorch/blob/main/pytorch01/pytorch04.ipynb)

[합성곱신경망](https://github.com/hugoMGSung/study-pytorch/blob/main/pytorch01/pytorch05.ipynb)

### 파이토치 신경망 훈련 순서
1. 데이터 준비
	- 데이터셋 로드: PyTorch의 Dataset 클래스를 활용하거나, torchvision 같은 라이브러리에서 제공하는 데이터셋을 사용합니다.
	- 데이터 전처리: 필요한 경우 데이터를 정규화, 원-핫 인코딩 등으로 변환합니다.
	- DataLoader 생성: PyTorch의 DataLoader를 사용하여 데이터를 배치로 나누고, 셔플 및 병렬 처리를 설정합니다.
2. 모델 정의
	- 신경망 설계: PyTorch의 torch.nn.Module을 상속받아 모델 구조를 정의합니다.
	
	```python
	class NeuralNet(nn.Module):
		def __init__(self):
			super(NeuralNet, self).__init__()
			self.fc = nn.Linear(784, 10)  # 예: 784 입력 -> 10 출력

		def forward(self, x):
			return self.fc(x)

	model = NeuralNet()
	```

3. 손실 함수와 옵티마이저 정의
	- 손실 함수: 문제 유형에 맞는 손실 함수 선택 (nn.CrossEntropyLoss, nn.MSELoss 등).
	- 옵티마이저: torch.optim에서 SGD, Adam 등 선택 후 모델 파라미터를 전달.

	```python
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	```

4. 훈련 루프 정의
	- 모델 훈련:
		- 데이터를 모델에 전달하고 예측값 생성.
		- 손실 계산.
		- optimizer.zero_grad()로 이전 기울기 초기화.
		- loss.backward()로 기울기 계산(역전파).
		- optimizer.step()으로 파라미터 업데이트.
	- 반복: 여러 에포크(epoch) 동안 데이터를 처리.

	```python
	for epoch in range(num_epochs):
		for inputs, labels in dataloader:
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

	```

5. 모델 평가
	- 훈련이 완료되면 검증 데이터셋 또는 테스트 데이터셋으로 성능 평가.
	- torch.no_grad() 블록 내에서 평가(기울기 계산 방지).

	```python
	with torch.no_grad():
		for inputs, labels in test_loader:
			outputs = model(inputs)
			# 예: 정확도 계산

	```

6. 모델 저장 및 불러오기
	- 저장: torch.save()로 모델 저장

	```python
	torch.save(model.state_dict(), 'model.pth')

	```

	- 불러오기: 저장된 모델 로드

	```python
	model.load_state_dict(torch.load('model.pth'))
	model.eval()
	```

### 파이토치 HS 학습

[링크](https://github.com/hugoMGSung/study-pytorch/tree/main/hansung_pytorch)

#### 선형회귀 부터 CNN 까지
- 완료

### 파이토치 실습

[링크](https://github.com/hugoMGSung/study-pytorch/tree/main/after_hs)

#### RNN 순환신경망

