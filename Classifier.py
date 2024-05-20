import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# 이미지 전처리 및 데이터셋 생성
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset = torchvision.datasets.ImageFolder(root='./FaceID/images', transform=transform)
print(trainset.__getitem__(0), len(trainset))
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# # 사전 훈련된 VGG 모델 불러오기
# vgg_model = models.vgg16(pretrained=True)
# for param in vgg_model.parameters():
#     param.requires_grad = False

# # 새로운 분류 레이어 추가
# num_features = vgg_model.classifier[6].in_features
# vgg_model.classifier[6] = nn.Sequential(
#     nn.Linear(num_features, 512),
#     nn.ReLU(),
#     nn.Dropout(0.5),
#     nn.Linear(512, 2),
#     nn.Softmax(dim=1)
# )

# # 모델을 GPU 또는 CPU로 이동
# vgg_model = vgg_model.to(device)

# # 손실 함수 및 옵티마이저 정의
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(vgg_model.parameters(), lr=0.001)

# # 모델 학습
# num_epochs = 10
# for epoch in range(num_epochs):
#     running_loss = 0.0
#     for images in dataloader:
#         images = images.to(device)
#         labels = torch.tensor([1] * images.size(0) if "positive" in image_paths else [0] * images.size(0)).to(device)

#         optimizer.zero_grad()

#         outputs = vgg_model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss}")

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
