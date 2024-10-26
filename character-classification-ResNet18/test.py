# test.py
import torch
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import json
import os

# 1. 載入 class_to_idx
with open('class_to_idx.json', 'r') as f:
    class_to_idx = json.load(f)

# 反轉字典，從索引找到角色名稱
idx_to_class = {v: k for k, v in class_to_idx.items()}

# 2. 載入模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18()
num_classes = len(idx_to_class)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load('character_classification_resnet18.pth'))
model = model.to(device)
model.eval()

# 3. 定義影像轉換
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. 預測單張圖片的函式
def predict_image(image_path, model, device):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

    return idx_to_class[predicted.item()]

# 5. 批次預測並生成 CSV
def generate_predictions_csv(model, test_folder, output_csv='predictions.csv'):
    results = []

    for img_name in os.listdir(test_folder):
        if img_name.endswith('.jpg') or img_name.endswith('.png'):
            img_path = os.path.join(test_folder, img_name)
            character = predict_image(img_path, model, device)
            results.append({'id': img_name.split('.')[0], 'character': character})

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f'預測結果已儲存至 {output_csv}')

# 6. 執行預測
test_folder = './test-final/test-final'  # 替換為你的測試資料夾
generate_predictions_csv(model, test_folder)
