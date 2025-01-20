import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from flask import Flask, request, jsonify
from prometheus_client import start_http_server, Summary
from PIL import Image
import io

# 데이터 전처리 변환
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# MNIST 데이터셋 로드
train_dataset = datasets.MNIST(root='./pytorch_level3/data', train=True, download=True, transform=transform)
val_dataset = datasets.MNIST(root='./pytorch_level3/data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 간단한 CNN 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 모델 학습
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

# 모델 저장
torch.save(model, './pytorch_level3/model.pth')

# Flask 앱 정의
app = Flask(__name__)

# Prometheus 설정
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')

# 모델 로드
model = torch.load('./pytorch_level3/model.pth')
model.eval()

# 이미지 전처리 함수
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

# 예측 함수
@REQUEST_TIME.time()
def get_prediction(image_bytes):
    tensor = transform_image(image_bytes)
    outputs = model(tensor)
    _, predicted = torch.max(outputs.data, 1)
    return predicted.item()

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        prediction = get_prediction(img_bytes)
        return jsonify({'prediction': prediction})

if __name__ == '__main__':
    start_http_server(8000)  # Prometheus HTTP 서버 시작
    app.run()