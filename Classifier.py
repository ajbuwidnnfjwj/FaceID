import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, transforms
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 이미지 전처리 및 데이터셋 생성
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset = torchvision.datasets.ImageFolder(root='./FaceID/images', transform=transform)
dataloader = DataLoader(trainset, batch_size=32, shuffle=True)

# 사전 훈련된 VGG 모델 불러오기
vgg_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
for param in vgg_model.parameters():
    param.requires_grad = False

# 새로운 분류 레이어 추가
num_features = vgg_model.classifier[6].in_features
vgg_model.classifier[6] = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 2),
    nn.Softmax(dim=1)
)

# 모델을 GPU 또는 CPU로 이동
vgg_model = vgg_model.to(device)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg_model.parameters(), lr=0.001)

# 모델 학습
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainset, 0):
        # 입력 데이터와 레이블을 GPU로 이동
        inputs, labels = data[0].transpose().to(device), torch.FloatTensor(data[1]).to(device)
        # 파라미터 그라디언트 초기화
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화
        outputs = vgg_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 통계 출력
        running_loss += loss.item()
        if i % 2000 == 1999:  # 매 2000 미니배치마다
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

# # 학습된 모델 저장
# torch.save(vgg_model.state_dict(), "face_classification_model.pth")

# # 새로운 이미지 분류 함수
# def classify_face(image_path):
#     image = Image.open(image_path).convert('RGB')
#     image = transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         output = vgg_model(image)
#     _, predicted = torch.max(output, 1)
#     if predicted.item() == 1:
#         return "얼굴입니다."
#     else:
#         return "얼굴이 아닙니다."

# # 새로운 이미지 분류 예시
# new_image_path = "path_to_new_image"
# result = classify_face(new_image_path)
# print("분류 결과:", result)
