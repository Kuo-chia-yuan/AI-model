import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer, DistilBertModel

# ========== STEP 1: 設置 GPU/CPU 裝置 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ========== STEP 2: 讀取測試資料 (test.json) ==========
try:
    test_data = pd.read_json('test.json')
except ValueError as e:
    print(f"Error reading JSON: {e}")
    exit()

print(f"Loaded {len(test_data)} rows from test.json.")

# 合併 title 和 text 作為輸入
test_data.fillna('', inplace=True)  # 避免 NaN 錯誤
test_data['input'] = test_data['title'] + ' ' + test_data['text']

# ========== STEP 3: 初始化 BERT Tokenizer ==========
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# 使用 tokenizer 將文本轉換為 BERT 的輸入格式
inputs = tokenizer(
    test_data['input'].tolist(),
    padding=True,
    truncation=True,
    return_tensors="pt",
    max_length=64
)

# 建立 DataLoader
X_input_ids = inputs['input_ids']
X_attention_mask = inputs['attention_mask']

test_dataset = TensorDataset(X_input_ids, X_attention_mask)
test_loader = DataLoader(test_dataset, batch_size=16)

# ========== STEP 4: 定義與載入已保存的模型 ==========
class BertClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(BertClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

# 初始化模型並載入權重
model = BertClassifier(num_classes=5).to(device)
model.load_state_dict(torch.load('bert_classifier.pth'))  # 載入已保存的模型
model.eval()  # 設置為評估模式

# ========== STEP 5: 進行推論 ==========
predictions = []
with torch.no_grad():
    for input_ids, attention_mask in test_loader:
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        outputs = model(input_ids, attention_mask)  # 前向傳播
        batch_predictions = torch.argmax(outputs, dim=1).cpu().numpy()  # 預測類別 (0-4)
        predictions.extend(batch_predictions + 1)  # 將類別轉回 1-5

# ========== STEP 6: 生成 CSV ==========
result_df = pd.DataFrame({
    'index': [f'index_{i}' for i in range(len(predictions))],
    'rating': predictions
})

# 將結果保存為 CSV
result_df.to_csv('predictions.csv', index=False)
print("Predictions saved to 'predictions.csv'.")
