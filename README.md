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

### 파이토치 신경망 훈련 순서
1. 데이터 준비 (Data Preparation)
	- 데이터셋 로드: 훈련과 검증을 위한 데이터를 로드합니다. 일반적으로 torchvision이나 torch.utils.data.DataLoader를 사용하여 데이터셋을 불러옵니다.
	- 데이터 전처리: 데이터를 모델에 맞게 변환합니다. 예를 들어, 이미지 데이터는 텐서로 변환하고, 필요하다면 정규화(normalization)나 크기 조정(resizing)을 합니다.
	- DataLoader 생성: PyTorch의 DataLoader를 사용하여 데이터를 배치로 나누고, 셔플 및 병렬 처리를 설정합니다.

	```python
	from torch.utils.data import DataLoader
	from torchvision import datasets, transforms

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5,), (0.5,))  # 정규화
	])
	train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
	train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
	```

2. 모델 정의 (Model Definition)
	- 모델을 정의합니다. 파이토치에서는 nn.Module을 상속받아 신경망 모델을 정의할 수 있습니다. 모델 구조에는 레이어, 활성화 함수, 순전파(forward) 방식이 포함됩니다.
	
	```python
	import torch
	import torch.nn as nn
	import torch.optim as optim

	class SimpleNN(nn.Module):
		def __init__(self):
			super(SimpleNN, self).__init__()
			self.fc1 = nn.Linear(28 * 28, 128)  # 입력 차원 28x28
			self.fc2 = nn.Linear(128, 10)  # 출력 차원 10 (MNIST 10개 클래스)

		def forward(self, x):
			x = x.view(-1, 28 * 28)  # Flatten
			x = torch.relu(self.fc1(x))  # 활성화 함수
			x = self.fc2(x)
			return x
	```

3. 손실 함수 정의 (Loss Function)
	- 모델이 예측한 값과 실제 값 사이의 차이를 계산할 손실 함수(loss function)를 정의합니다. 예를 들어, 분류 문제의 경우 CrossEntropyLoss를 사용할 수 있습니다.


	```python
	loss_fn = nn.CrossEntropyLoss()  # 다중 클래스 분류 손실 함수
	```

4. 옵티마이저 정의 (Optimizer Definition)
	- 모델의 파라미터를 최적화할 옵티마이저를 정의합니다. 대표적인 옵티마이저로는 SGD, Adam 등이 있습니다.

	```python
	optimizer = optim.Adam(model.parameters(), lr=0.001)

	```

5. 훈련 루프 (Training Loop)
	- 데이터셋에 대해 여러 번 학습을 반복합니다. 매 반복마다 데이터를 모델에 입력하고, 예측 값을 얻은 후, 손실을 계산하고, 역전파(backpropagation)를 통해 가중치를 업데이트합니다.
	- 모델 훈련:
		- 데이터를 모델에 전달하고 예측값 생성.
		- 손실 계산.
		- optimizer.zero_grad()로 이전 기울기 초기화.
		- loss.backward()로 기울기 계산(역전파).
		- optimizer.step()으로 파라미터 업데이트.
	- 반복: 여러 에포크(epoch) 동안 데이터를 처리.

	```python
	for epoch in range(10):  # 10번 에폭을 훈련
		for data, target in train_loader:
			optimizer.zero_grad()  # 이전 기울기 초기화
			output = model(data)  # 모델 예측
			loss = loss_fn(output, target)  # 손실 계산
			loss.backward()  # 역전파
			optimizer.step()  # 가중치 업데이트
		
		print(f'Epoch {epoch+1}, Loss: {loss.item()}')

	```

6. 모델 평가 (Model Evaluation)
	- 모델이 얼마나 잘 학습되었는지 평가하기 위해, 검증 데이터셋을 사용하여 예측을 실행하고 정확도 등의 평가 지표를 계산합니다.
	- 훈련이 완료되면 검증 데이터셋 또는 테스트 데이터셋으로 성능 평가.
	- torch.no_grad() 블록 내에서 평가(기울기 계산 방지).

	```python
	model.eval()  # 평가 모드로 전환
	correct = 0
	total = 0
	with torch.no_grad():  # 기울기 계산 비활성화
		for data, target in test_loader:
			output = model(data)
			_, predicted = torch.max(output, 1)
			total += target.size(0)
			correct += (predicted == target).sum().item()

	print(f'Accuracy: {100 * correct / total}%')
	```

7. 모델 저장 및 불러오기 (Save and Load Model)
	- 훈련이 끝난 후, 모델을 저장하고, 나중에 불러와서 사용할 수 있습니다.
	- 저장: torch.save()로 모델 저장

	```python
	# 모델 저장
	torch.save(model.state_dict(), 'model.pth')

	# 모델 불러오기
	model = SimpleNN()
	model.load_state_dict(torch.load('model.pth'))

	```

8. 대부분(!)의 딥러닝은 이 순서를 따릅니다. 

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


### 3분 딥러닝 따라만하기
- https://github.com/keon/3-min-pytorch/tree/master

[링크](https://github.com/hugoMGSung/study-pytorch/tree/main/3min_pytorch)


### 파이토치 HS 학습

[링크](https://github.com/hugoMGSung/study-pytorch/tree/main/hansung_pytorch)

#### 선형회귀 부터 CNN 까지
- 완료

### 파이토치 실습

[링크](https://github.com/hugoMGSung/study-pytorch/tree/main/after_hs)

### 캐글 Kagglehub 사용

- 캐글 로그인 후 셋팅에서 API 토큰 생성, kaggle.json 파일 다운로드

	<img src="https://raw.githubusercontent.com/hugoMGSung/study-pytorch/main/images/torch0020.png" width="730">




