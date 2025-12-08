# 辛普森家庭影像辨識 - 三分割版本（含解壓縮）
# ============================================
# 1. 安裝和導入必要的庫
# ============================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import zipfile
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 檢查 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用設備: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ============================================
# 2. 解壓縮 train.zip
# ============================================
print("\n" + "="*50)
print("步驟 1：解壓縮資料集")
print("="*50)

# 檢查 train.zip 是否存在
zip_path = '/content/train.zip'

if not os.path.exists(zip_path):
    print(f"❌ 找不到 {zip_path}")
    print("\n請確認 train.zip 已上傳到 Colab")
    print("\n可能的位置：")
    print("1. /content/train.zip")
    print("2. /content/drive/MyDrive/train.zip (如果在 Google Drive)")

    # 列出 /content 目錄內容
    print("\n當前 /content 目錄內容：")
    print(os.listdir('/content'))

    # 如果需要從 Google Drive 讀取，取消下面註解
    # from google.colab import drive
    # drive.mount('/content/drive')
    # zip_path = '/content/drive/MyDrive/train.zip'

    raise FileNotFoundError(f"找不到 {zip_path}，請先上傳 train.zip")
else:
    print(f"✅ 找到 train.zip: {zip_path}")

    # 檢查檔案大小
    file_size = os.path.getsize(zip_path) / (1024 * 1024)  # MB
    print(f"檔案大小: {file_size:.2f} MB")

# 解壓縮
print("\n開始解壓縮...")
extract_dir = '/content/data'

try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # 顯示進度
        file_list = zip_ref.namelist()
        print(f"壓縮檔內含 {len(file_list)} 個檔案")

        # 解壓縮
        for file in tqdm(file_list, desc='解壓縮中'):
            zip_ref.extract(file, extract_dir)

    print("✅ 解壓縮完成！")
except Exception as e:
    print(f"❌ 解壓縮失敗: {e}")
    raise

# 顯示解壓後的目錄結構
print("\n解壓後的目錄結構：")
for root, dirs, files in os.walk(extract_dir):
    level = root.replace(extract_dir, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    if level < 2:  # 只顯示前兩層
        subindent = ' ' * 2 * (level + 1)
        for d in dirs[:5]:  # 只顯示前5個資料夾
            print(f'{subindent}{d}/')
        if len(dirs) > 5:
            print(f'{subindent}... (還有 {len(dirs)-5} 個資料夾)')
        break  # 只顯示第一層

# ============================================
# 3. 路徑設定
# ============================================
print("\n" + "="*50)
print("步驟 2：設定資料路徑")
print("="*50)

data_dir = '/content/data/train'

# 檢查路徑是否存在
if not os.path.exists(data_dir):
    print(f"❌ 找不到路徑: {data_dir}")
    print("\n正在搜尋可能的路徑...")

    possible_paths = [
        '/content/data/train',
        '/content/train',
        '/content/data/Train',
        '/content/data'
    ]

    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            try:
                subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                if len(subdirs) > 10:
                    data_dir = path
                    print(f"✅ 找到有效路徑: {data_dir}")
                    break
            except:
                continue

    if not os.path.exists(data_dir):
        raise FileNotFoundError("請檢查資料路徑")

print(f"\n✅ 資料路徑: {data_dir}")

# 獲取所有角色資料夾
classes = sorted(os.listdir(data_dir))
classes = [c for c in classes if os.path.isdir(os.path.join(data_dir, c))]
print(f"角色數量: {len(classes)}")
print(f"前10個角色: {classes[:10]}")

# 統計每個類別的圖片數量
print("\n每個類別的圖片數量：")
total_images = 0
class_image_counts = {}

for cls in classes:
    cls_path = os.path.join(data_dir, cls)
    num_images = len([f for f in os.listdir(cls_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))])
    class_image_counts[cls] = num_images
    total_images += num_images

# 顯示前5個類別
for cls in classes[:5]:
    print(f"  {cls}: {class_image_counts[cls]} 張")

if len(classes) > 5:
    print("  ...")

print(f"\n總圖片數量: {total_images} 張")
print(f"平均每個角色: {total_images // len(classes)} 張")
print(f"最多圖片的角色: {max(class_image_counts, key=class_image_counts.get)} ({max(class_image_counts.values())} 張)")
print(f"最少圖片的角色: {min(class_image_counts, key=class_image_counts.get)} ({min(class_image_counts.values())} 張)")

# ============================================
# 4. 數據轉換和增強
# ============================================
print("\n" + "="*50)
print("步驟 3：設定數據轉換")
print("="*50)

# 訓練集轉換（有增強）
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# 驗證集和測試集轉換（無增強）
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

print("✅ 訓練集轉換：Resize(256) → RandomCrop(224) → Flip → ColorJitter → Normalize")
print("✅ 驗證/測試集轉換：Resize(224) → Normalize")

# ============================================
# 5. 載入並分割數據集（Train/Val/Test）
# ============================================
print("\n" + "="*50)
print("步驟 4：載入並分割數據集")
print("="*50)

# 先載入完整數據集
full_dataset = datasets.ImageFolder(root=data_dir)

# 計算分割大小：70% 訓練，15% 驗證，15% 測試
total_size = len(full_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

print(f"\n數據集分割:")
print(f"總計: {total_size}")
print(f"訓練集: {train_size} ({train_size/total_size*100:.1f}%)")
print(f"驗證集: {val_size} ({val_size/total_size*100:.1f}%)")
print(f"測試集: {test_size} ({test_size/total_size*100:.1f}%)")

# 分割數據集
train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# 應用不同的轉換
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = eval_transform
test_dataset.dataset.transform = eval_transform

# 創建 DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                         shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size,
                       shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=2, pin_memory=True)

# 獲取類別名稱
class_names = full_dataset.classes
num_classes = len(class_names)
print(f"\n類別數量: {num_classes}")
print(f"Batch size: {batch_size}")
print(f"訓練 batches: {len(train_loader)}")
print(f"驗證 batches: {len(val_loader)}")
print(f"測試 batches: {len(test_loader)}")

# ============================================
# 6. 可視化部分樣本
# ============================================
print("\n" + "="*50)
print("步驟 5：可視化樣本")
print("="*50)

def imshow(img, title):
    """反標準化並顯示圖片"""
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')

# 顯示一批樣本
dataiter = iter(train_loader)
images, labels = next(dataiter)

plt.figure(figsize=(15, 8))
for idx in range(min(8, len(images))):
    plt.subplot(2, 4, idx + 1)
    imshow(images[idx], class_names[labels[idx]])
plt.tight_layout()
plt.show()

# ============================================
# 7. 構建 ResNet-50 模型
# ============================================
print("\n" + "="*50)
print("步驟 6：構建模型")
print("="*50)

def create_model(num_classes, pretrained=True):
    """創建 ResNet-50 模型"""
    model = models.resnet50(pretrained=pretrained)

    # 凍結所有層
    for param in model.parameters():
        param.requires_grad = False

    # 替換分類頭
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

    return model

model = create_model(num_classes).to(device)

print("\n模型架構:")
print(model.fc)

# 計算參數量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n總參數量: {total_params:,}")
print(f"可訓練參數: {trainable_params:,}")
print(f"凍結參數: {total_params - trainable_params:,}")

# ============================================
# 8. 定義損失函數和優化器
# ============================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

print(f"\n損失函數: CrossEntropyLoss")
print(f"優化器: Adam (lr=0.001)")
print(f"學習率調度: StepLR (step_size=7, gamma=0.1)")

# ============================================
# 9. 訓練函數
# ============================================
def train_epoch(model, loader, criterion, optimizer, device):
    """訓練一個 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='訓練中')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# ============================================
# 10. 驗證/測試函數
# ============================================
def evaluate(model, loader, criterion, device, desc='評估中'):
    """評估模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(loader, desc=desc)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

# ============================================
# 11. 訓練循環 - 階段一：只訓練分類頭
# ============================================
print("\n" + "="*50)
print("階段一：訓練分類頭（凍結 ResNet-50 骨幹）")
print("="*50)

num_epochs_phase1 = 10
best_val_acc = 0.0
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
}

for epoch in range(num_epochs_phase1):
    print(f'\nEpoch {epoch+1}/{num_epochs_phase1}')
    print('-' * 50)

    # 訓練
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

    # 驗證
    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device, desc='驗證中')

    # 更新學習率
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    # 記錄歷史
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), '/content/best_model_phase1.pth')
        print(f'✅ 保存最佳模型！驗證準確率: {val_acc:.2f}%')

    print(f'訓練 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
    print(f'驗證 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
    print(f'學習率: {current_lr:.6f}')

# ============================================
# 12. 訓練循環 - 階段二：微調整個網路
# ============================================
print("\n" + "="*50)
print("階段二：微調整個網路")
print("="*50)

# 載入階段一的最佳模型
model.load_state_dict(torch.load('/content/best_model_phase1.pth'))

# 解凍所有層
for param in model.parameters():
    param.requires_grad = True

# 計算可訓練參數
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"可訓練參數: {trainable_params:,}")

# 使用較小的學習率
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

num_epochs_phase2 = 15

for epoch in range(num_epochs_phase2):
    print(f'\nEpoch {epoch+1}/{num_epochs_phase2}')
    print('-' * 50)

    # 訓練
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

    # 驗證
    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device, desc='驗證中')

    # 更新學習率
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    # 記錄歷史
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    # 保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), '/content/best_model_final.pth')
        print(f'✅ 保存最佳模型！驗證準確率: {val_acc:.2f}%')

    print(f'訓練 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
    print(f'驗證 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
    print(f'學習率: {current_lr:.6f}')

# ============================================
# 13. 繪製訓練曲線
# ============================================
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.axvline(x=num_epochs_phase1-0.5, color='r', linestyle='--', label='Phase 2 Start')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.axvline(x=num_epochs_phase1-0.5, color='r', linestyle='--', label='Phase 2 Start')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('/content/training_history.png', dpi=300, bbox_inches='tight')
plt.show()

print(f'\n🎉 訓練完成！最佳驗證準確率: {best_val_acc:.2f}%')

# ============================================
# 14. 在測試集上評估（最終評估）
# ============================================
print("\n" + "="*50)
print("最終評估：在測試集上測試")
print("="*50)

# 載入最佳模型
model.load_state_dict(torch.load('/content/best_model_final.pth'))

# 在測試集上評估
test_loss, test_acc, test_preds, test_labels = evaluate(
    model, test_loader, criterion, device, desc='測試中'
)

print(f'\n📊 最終結果:')
print(f'訓練集最佳準確率: {history["train_acc"][-1]:.2f}%')
print(f'驗證集最佳準確率: {best_val_acc:.2f}%')
print(f'測試集準確率: {test_acc:.2f}% ⭐')
print(f'測試集損失: {test_loss:.4f}')

# ============================================
# 15. 測試集混淆矩陣
# ============================================
cm = confusion_matrix(test_labels, test_preds)

plt.figure(figsize=(20, 18))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('預測類別')
plt.ylabel('真實類別')
plt.title('測試集混淆矩陣')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('/content/test_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 16. 測試集分類報告
# ============================================
print("\n測試集分類報告:")
print(classification_report(test_labels, test_preds,
                          target_names=class_names,
                          digits=3))

# ============================================
# 17. 找出測試集中最容易混淆的角色對
# ============================================
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
np.fill_diagonal(cm_normalized, 0)

confusion_pairs = []
for i in range(len(class_names)):
    for j in range(len(class_names)):
        if i != j and cm[i, j] > 0:
            confusion_pairs.append((
                class_names[i],
                class_names[j],
                cm[i, j],
                cm_normalized[i, j]
            ))

confusion_pairs.sort(key=lambda x: x[2], reverse=True)

print("\n測試集中最容易混淆的角色對 (Top 10):")
print("-" * 80)
for idx, (true_class, pred_class, count, ratio) in enumerate(confusion_pairs[:10], 1):
    print(f"{idx}. {true_class:20s} → {pred_class:20s} | 錯誤次數: {count:3d} ({ratio*100:.1f}%)")

# ============================================
# 18. 保存完整結果
# ============================================
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': class_names,
    'num_classes': num_classes,
    'best_val_acc': best_val_acc,
    'test_acc': test_acc,
    'test_loss': test_loss,
    'history': history
}, '/content/simpson_classifier_complete.pth')

print("\n✅ 模型已保存為 '/content/simpson_classifier_complete.pth'")

# ============================================
# 19. 預測函數（用於新圖片）
# ============================================
def predict_image(image_path, model, transform, class_names, device):
    """預測單張圖片"""
    from PIL import Image

    model.eval()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top5_prob, top5_idx = torch.topk(probabilities, 5)

    print(f"\n預測結果 (Top 5):")
    for i in range(5):
        class_name = class_names[top5_idx[0][i]]
        prob = top5_prob[0][i].item() * 100
        print(f"{i+1}. {class_name:20s}: {prob:.2f}%")

    return class_names[top5_idx[0][0]]

# ============================================
# 20. 總結報告
# ============================================
print("\n" + "="*50)
print("🎊 訓練完成！最終報告")
print("="*50)
print(f"\n數據集分割:")
print(f"  訓練集: {train_size} 張 (70%)")
print(f"  驗證集: {val_size} 張 (15%)")
print(f"  測試集: {test_size} 張 (15%)")
print(f"\n模型性能:")
print(f"  驗證集最佳準確率: {best_val_acc:.2f}%")
print(f"  測試集準確率: {test_acc:.2f}% ⭐")
print(f"  差距: {abs(best_val_acc - test_acc):.2f}%")

if abs(best_val_acc - test_acc) < 2.0:
    print("\n✅ 模型泛化能力極佳！驗證集和測試集性能接近")
elif abs(best_val_acc - test_acc) < 5.0:
    print("\n✅ 模型泛化能力良好")
else:
    print("\n⚠️ 驗證集和測試集差距較大，可能存在輕微過擬合")

print("\n保存的文件:")
print("  - best_model_final.pth (最佳模型權重)")
print("  - simpson_classifier_complete.pth (完整模型信息)")
print("  - training_history.png (訓練曲線)")
print("  - test_confusion_matrix.png (測試集混淆矩陣)")

# ============================================
# 21. 下載結果文件（可選）
# ============================================
print("\n如需下載結果，執行以下代碼：")
print("from google.colab import files")
print("files.download('/content/best_model_final.pth')")
print("files.download('/content/simpson_classifier_complete.pth')")
print("files.download('/content/training_history.png')")
print("files.download('/content/test_confusion_matrix.png')")
