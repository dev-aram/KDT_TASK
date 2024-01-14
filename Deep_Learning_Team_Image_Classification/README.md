# KDT 2차 프로젝트 팀과제
## 딥러닝- 이미지 분류

<br>

👀**주제Weather Image Recognition**<br>
https://www.kaggle.com/datasets/jehanbhathena/weather-dataset

---

- 필요한 데이터를 케글에서 AIP 형식으로 받아온 후 이미지 train, validation으로 분류 작업 진행
``` python
import os
os.environ['KAGGLE_USERNAME'] = '**'
os.environ['KAGGLE_KEY'] = '**'

!kaggle datasets download -d jehanbhathena/weather-dataset
!unzip -q weather-dataset.zip
```
``` python
import random
import shutil

def split_dataset_by_class(dataset_path, train_path, validation_path, validation_ratio=0.2):
    # 클래스 폴더 목록 가져오기
    classes = os.listdir(dataset_path)

    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        train_class_path = os.path.join(train_path, class_name)
        validation_class_path = os.path.join(validation_path, class_name)

        # 폴더 생성
        if not os.path.exists(train_class_path):
            os.makedirs(train_class_path)
        if not os.path.exists(validation_class_path):
            os.makedirs(validation_class_path)

        # 클래스 폴더 내의 파일 목록 가져오기
        file_list = os.listdir(class_path)

        # 클래스 별 데이터셋 섞기
        random.shuffle(file_list)

        # 클래스 별 데이터셋을 train과 validation으로 나누기
        num_validation = int(len(file_list) * validation_ratio)
        validation_files = file_list[:num_validation]
        train_files = file_list[num_validation:]

        # validation 폴더로 파일 이동
        for file in validation_files:
            src_path = os.path.join(class_path, file)
            dest_path = os.path.join(validation_class_path, file)
            shutil.move(src_path, dest_path)

        # train 폴더로 파일 이동
        for file in train_files:
            src_path = os.path.join(class_path, file)
            dest_path = os.path.join(train_class_path, file)
            shutil.move(src_path, dest_path)

# 사용 예시
dataset_path = 'dataset'
train_path = 'train'
validation_path = 'validation'

split_dataset_by_class(dataset_path, train_path, validation_path, validation_ratio=0.2)
```
위 코드대로 작성 시 아래 이미지처럼 dataset이 train, validation으로 폴더가 분류되고 이미지가 옮겨짐
![image](https://github.com/dev-aram/KDT_TASK/assets/135501045/1215f935-8b6a-47e3-ab11-cded6fa9e10e)

---

### 필요한 도구 임포트
필요한 라이브러리들을 import

``` python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
```

### GPU 확인
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
```

### 데이터 변형
데이터 학습 시 오버피팅을 막기 위해 일부 데이터 변형 작업 진행
```python
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]),
    'validation': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
}

def target_transforms(target):
    return torch.FloatTensor([target])
```

### ImageFolaer 함수를 이용해서 이미지를 불러오기
> 라벨을 float 타임으로 바꿔줄 필요가 없기때문에 target_transform은 제외

```python
image_datasets = {
    'train': datasets.ImageFolder('train', data_transforms['train']),
    'validation': datasets.ImageFolder('validation', data_transforms['validation'])
}
```

### 데이터 로더 생성
```python
dataloaders = {
    'train': DataLoader(
        image_datasets['train'],
        batch_size=32,
        shuffle=True
    ),
    'validation': DataLoader(
        image_datasets['validation'],
        batch_size=32,
        shuffle=False
    )
}
```

### 이미지 subplots으로 시각화
```python
imgs, labels = next(iter(dataloaders['train']))
fig, axes = plt.subplots(4, 8, figsize=(16, 8))

for ax, img, label in zip(axes.flatten(), imgs, labels):
    ax.imshow(img.permute(1, 2, 0))
    ax.set_title(label.item())
    ax.axis('off')
```
![image](https://github.com/dev-aram/KDT_TASK/assets/135501045/26c0dbca-c853-4c9e-bf00-ec37e8568d97)

---
### 학습
#### 모델 선정(MobileNet V2)
MobileNet V2
Google은 2018년 MobileNet V2를 제안한 논문인 MobileNetV2: Inverted Residuals and Linear Bottlenecks를 발표했습니다.

MobileNet V2는 이전 모델인 MobileNet을 개선한 네트워크 입니다. 따라서 MobileNet과 동일하게 MobileNet V2는 임베디드 디바이스 또는 모바일 장치를 타겟으로 하는 단순한 구조의 경량화 네트워크를 설계하는데 초점이 맞춰져 있습니다.

MobileNet V2는 MobileNet V1을 기반으로 두고 몇가지 개선점을 추가했습니다. MobileNet V1에서 사용하던 Depthwise-Separable Convolution을 주로 사용하고 width/resolution multiplyer를 사용해 정확도와 모델 크기를 trade-off하는 등 유사한 점이 많습니다.

하지만 단순히 Depthwise-Separable Convolution을 쌓은 구조의 MobileNet과는 달리 MobileNet v2에서는 Inverted Residual block이라는 구조를 이용해 네트워크를 구성한 차이점이 있습니다.

✨ 참조 https://velog.io/@woojinn8/LightWeight-Deep-Learning-7.-MobileNet-v2

---

#### 사용방법
```python
model = models.mobilenet_v2(pretrained=True).to(device)
model
```
**변경 전 classifier**
![image](https://github.com/dev-aram/KDT_TASK/assets/135501045/8b73cbcc-3b16-40d1-af93-9a4397304f84)

**내 데이터에 맞게 모델 변경**
```python
for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Linear(1280, 512),
    nn.ReLU(),
    nn.Linear(512, 128),
    nn.Sigmoid(),
    nn.Linear(128, 11)
).to(device)

print(model)
```
**변경 후 classifier**
![image](https://github.com/dev-aram/KDT_TASK/assets/135501045/ebcc3d6e-a705-4486-8d88-7e31d059e3c6)

---

#### 학습 진행
```python
# 학습
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

epochs = 10

for epoch in  range(epochs + 1):
    for phase in ['train', 'validation']:
        if phase == 'train':
            model.train()
        else:
            model.eval()
        sum_losses = 0
        sum_accs = 0

        for x_batch, y_batch in dataloaders[phase]:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            y_pred = model(x_batch)
            loss = nn.CrossEntropyLoss()(y_pred, y_batch)
            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            sum_losses = sum_losses + loss
            y_prob = nn.Softmax(1)(y_pred)
            y_pred_index = torch.argmax(y_prob, axis=1)

            acc = (y_batch == y_pred_index).float().sum() / len(y_batch) * 100
            sum_accs = sum_accs + acc

        avg_loss = sum_losses / len(dataloaders[phase])
        avg_acc = sum_accs / len(dataloaders[phase])
        print(f'{phase:10s}: Epoch {epoch+1:4d}/{epochs} Loss: {avg_loss:.4f} Accuracy: {avg_acc: .2f}%')

```

> train     : Epoch   11/10 Loss: 0.3118 Accuracy:  89.14%
> validation: Epoch   11/10 Loss: 0.3706 Accuracy:  87.09%

학습률이 낮게나와 모델 저장 후 다시 학습 진행

#### 모델 저장
```python
torch.save(model.state_dict(),'model.pth') # model.h5 (텐서플로우)
```

---

```python
for epoch in  range(epochs + 1):
    for phase in ['train', 'validation']:
        if phase == 'train':
            model.train()
        else:
            model.eval()
        sum_losses = 0
        sum_accs = 0

        for x_batch, y_batch in dataloaders[phase]:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            y_pred = model(x_batch)
            loss = nn.CrossEntropyLoss()(y_pred, y_batch)
            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            sum_losses = sum_losses + loss
            y_prob = nn.Softmax(1)(y_pred)
            y_pred_index = torch.argmax(y_prob, axis=1)

            acc = (y_batch == y_pred_index).float().sum() / len(y_batch) * 100
            sum_accs = sum_accs + acc

        avg_loss = sum_losses / len(dataloaders[phase])
        avg_acc = sum_accs / len(dataloaders[phase])
        print(f'{phase:10s}: Epoch {epoch+1:4d}/{epochs} Loss: {avg_loss:.4f} Accuracy: {avg_acc: .2f}%')

```
> train     : Epoch   11/10 Loss: 0.1785 Accuracy:  93.69%
> validation: Epoch   11/10 Loss: 0.4038 Accuracy:  87.67%
> 학습률과 정확도가 상승

---

## 테스트
만든 모델로 테스트 진행

```python
# 테스트
from PIL import Image
img1 = Image.open('./validation/dew/2264.jpg')
img2 = Image.open('./validation/frost/3676.jpg')

fig, axes = plt.subplots(1,  2, figsize=(12,  6))
axes[0].imshow(img1)
axes[0].axis('off')
axes[1].imshow(img2)
axes[1].axis('off')
plt.show()
```
![image](https://github.com/dev-aram/KDT_TASK/assets/135501045/983cd66a-8a11-4950-a190-4d8a466d7303)

> 위 이미지를 가지고 테스트를 진행

```python
img1_input = data_transforms['validation'](img1)
img2_input = data_transforms['validation'](img2)
print(img1_input.shape)
print(img2_input.shape)

test_batch = torch.stack([img1_input, img2_input])
test_batch = test_batch.to(device)
test_batch.shape
y_pred = model(test_batch)
y_pred
```

```python
# 강사님 코드
y_prob = nn.Softmax(1)(y_pred)
probs, idx = torch.topk(y_prob, k=3)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].set_title('{:.2f}% {}, {:.2f}% {}, {:.2f}% {}'.format(
    probs[0, 0] * 100,
    image_datasets['validation'].classes[idx[0, 0]],
    probs[0, 1] * 100,
    image_datasets['validation'].classes[idx[0, 1]],
    probs[0, 2] * 100,
    image_datasets['validation'].classes[idx[0, 2]],
))
axes[0].imshow(img1)
axes[0].axis('off')
axes[1].set_title('{:.2f}% {}, {:.2f}% {}, {:.2f}% {}'.format(
    probs[1, 0] * 100,
    image_datasets['validation'].classes[idx[1, 0]],
    probs[1, 1] * 100,
    image_datasets['validation'].classes[idx[1, 1]],
    probs[1, 2] * 100,
    image_datasets['validation'].classes[idx[1, 2]],
))
axes[1].imshow(img2)
axes[1].axis('off')
plt.show()
```

![image](https://github.com/dev-aram/KDT_TASK/assets/135501045/d5a5dbf7-7a11-441f-9bd0-3044741d5770)

위 모델로 테스트 결과 맞게 나온것을 확인.
