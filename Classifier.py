import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, transforms
from torch.utils.data import DataLoader
import cv2 as cv
from PIL import Image
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Classifier():
    def __init__(self) -> None:
        self.face_classifier = models.vgg16(pretrained=False)
        for param in self.face_classifier.parameters(): #합성곱 레이어는 초기화하지 않음
                param.requires_grad = False
        num_features = self.face_classifier.classifier[6].in_features #fully connected layer만 새로 바꿔줌
        self.face_classifier.classifier[6] = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        )
        self.face_classifier.to(device)
        self.transform = None
        try:
            self.face_classifier.load_state_dict(torch.load('vgg16_cifar10.pth'))
        except:
            self.pre_trained = False

            self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
            


    def classifyFace(self, path, file_name):
        self.trainAndSaveModel()
        image_name = os.path.join(path, file_name)
        image = cv.imread(image_name)
        
        assert image is not None, 'cannot load image'

        image = Image.fromarray(image)
        image = self.transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = self.face_classifier(image)
        _, predicted = torch.max(output, 1)
        print(image_name, end = ' ')
        if predicted.item() == 0:
            print("같은 얼굴입니다.")
        else:
            print("같은 얼굴이 아닙니다.")

    def classifyFace(self, image):        
        assert image is not None, 'cannot load image'

        image = Image.fromarray(image)
        image = self.transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = self.face_classifier(image)
        _, predicted = torch.max(output, 1)
        if predicted.item() == 0:
            print("같은 얼굴입니다.")
        else:
            print("같은 얼굴이 아닙니다.")


    def trainAndSaveModel(self):        
        # 이미지 전처리 및 데이터셋 생성
        trainset = torchvision.datasets.ImageFolder(root='./FaceID/images', transform=self.transform)
        trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.face_classifier.parameters(), lr=0.001)

        self.face_classifier.to(device)

        # 모델 학습
        num_epochs = 100
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = self.face_classifier(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 50 == 49: 
                    print("=============================================")
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 50))
                    print("=============================================")
                    running_loss = 0.0
        
        # 학습된 모델 저장
        torch.save(self.face_classifier.state_dict(), "face_classification_model.pth")
        self.pre_trained = True


    def evaluateModel(self):
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        self.face_classifier.eval()  # 평가 모드 설정
        all_preds = []
        all_labels = []

        dataset = torchvision.datasets.ImageFolder(root='./FaceID/images', transform=self.transform)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.face_classifier(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")

if __name__ == '__main__':
    command = input()
    if command == 'compile':
        pass
    elif command == 'test':
        classifier = Classifier()
        classifier.evaluateModel()
    else:
        classifier = Classifier()
        classifier.classifyFace('./FaceID/images/1','Liz_1.jpg')