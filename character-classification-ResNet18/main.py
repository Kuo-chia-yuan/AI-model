# train.py
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import json
import random
import numpy as np
from PIL import Image

# 1. 定義資料增強的轉換 (含旋轉、填充與雜訊)
def add_random_noise(img):
    """對輸入影像加入隨機雜訊。"""
    np_img = np.array(img)
    noise = np.random.normal(0, 25, np_img.shape).astype(np.uint8)  # 高斯雜訊
    noisy_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 調整尺寸
    transforms.RandomApply([transforms.RandomRotation(30)], p=0.5),  # 隨機旋轉
    transforms.RandomApply([transforms.Lambda(add_random_noise)], p=0.5),  # 加入雜訊
    transforms.RandomHorizontalFlip(p=0.5),  # 隨機水平翻轉
    transforms.Pad(padding=10, fill=0, padding_mode='constant'),  # 隨機填充黑邊
    transforms.RandomCrop(224),  # 裁剪回原尺寸
    transforms.ToTensor(),  # 轉換為 Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 正規化
])

# 2. 載入資料集 (ImageFolder 格式)
data_path = './train/train'
full_dataset = datasets.ImageFolder(data_path, transform=transform)

# 3. 將資料集分成訓練集和驗證集
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 4. 建立 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 5. 建立 ResNet 模型與優化器 (與之前相同)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=True)

# 凍結卷積層
for param in model.parameters():
    param.requires_grad = False

num_classes = len(full_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 6. 訓練與驗證函式 (略同於之前)
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        print(f'Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}')

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct / total
    print(f'Val Loss: {epoch_loss:.4f}, Val Acc: {epoch_acc:.4f}')

# 7. 訓練模型
num_epochs = 1
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    train(model, train_loader, criterion, optimizer, device)
    validate(model, val_loader, criterion, device)

# 8. 儲存模型與類別索引
torch.save(model.state_dict(), 'character_classification_resnet18.pth')

with open('class_to_idx.json', 'w') as f:
    json.dump(full_dataset.class_to_idx, f)

print('模型與類別索引已保存！')
