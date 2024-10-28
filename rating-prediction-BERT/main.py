import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split

# ========== STEP 1: 設置 GPU/CPU 裝置 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 清空 GPU 緩存
torch.cuda.empty_cache()

# ========== STEP 2: 讀取訓練資料 (train.json) ==========
try:
    data = pd.read_json('train.json')
except ValueError as e:
    print(f"Error reading JSON: {e}")
    exit()

print(f"Loaded {len(data)} rows.")
data = data[['title', 'text', 'rating']]

# 將 rating 減 1，讓其範圍變成 [0, 4]，符合 CrossEntropyLoss 的類別需求
data['rating'] = data['rating'] - 1

# 填補 NaN 值，避免型態錯誤
data.fillna('', inplace=True)

# ========== STEP 3: 初始化 BERT 模型和 Tokenizer ==========
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# 定義 BERT 分類模型
class BertClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(BertClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token 的嵌入
        return self.classifier(cls_output)

# 初始化模型並移至 GPU
model = BertClassifier(num_classes=5).to(device)

# ========== STEP 4: 資料預處理 ==========
# 將 title 和 text 合併為一個輸入
data['input'] = data['title'] + ' ' + data['text']

# 使用 tokenizer 將文本轉為 BERT 的輸入格式
inputs = tokenizer(
    data['input'].tolist(),
    padding=True,
    truncation=True,
    return_tensors="pt",
    max_length=64
)

# 將資料轉為 PyTorch Tensor
X_input_ids = inputs['input_ids']
X_attention_mask = inputs['attention_mask']
y = torch.tensor(data['rating'].values, dtype=torch.long)

# 切分訓練集和測試集
X_train_ids, X_test_ids, X_train_mask, X_test_mask, y_train, y_test = train_test_split(
    X_input_ids, X_attention_mask, y, test_size=0.2, random_state=42
)

# 建立 DataLoader
train_dataset = TensorDataset(X_train_ids, X_train_mask, y_train)
test_dataset = TensorDataset(X_test_ids, X_test_mask, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# ========== STEP 5: 設定損失函數和優化器 ==========
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# ========== STEP 6: 訓練模型 ==========
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
        input_ids, attention_mask, labels = (
            input_ids.to(device),
            attention_mask.to(device),
            labels.to(device)
        )

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Completed. Average Loss: {total_loss / len(train_loader):.4f}")

# ========== STEP 7: 保存模型 ==========
model_path = "bert_classifier.pth"
torch.save(model.state_dict(), model_path)  # 保存模型權重
print(f"Model saved to {model_path}")
