import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import Dataset

# 讀取 Excel 檔案
df = pd.read_excel('AI.xlsx')

# 將文章、問題及四個選項欄位合併成一個「輸入特徵」欄位
df['輸入特徵'] = (
    df['文章'].astype(str) + ' ' +
    df['問題'].astype(str) + ' ' +
    df['選項1'].astype(str) + ' ' +
    df['選項2'].astype(str) + ' ' +
    df['選項3'].astype(str) + ' ' +
    df['選項4'].astype(str)
)

# 初始化 BERT Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('DistilBert-base-uncased')

# 將資料 Tokenize
def encode_texts(texts):
    return tokenizer(
        texts,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

X_encodings = encode_texts(df['輸入特徵'].tolist())
y_tensor = torch.tensor(df['正確答案'].values) - 1  # 調整範圍為 0~3

# 建立自定義 Dataset 類別
class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

# 建立 Dataset
dataset = CustomDataset(X_encodings, y_tensor)

from transformers import Trainer, TrainingArguments

# 初始化 BERT 模型
model = DistilBertForSequenceClassification.from_pretrained('DistilBert-base-uncased', num_labels=4)

# 設定訓練參數
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=15,  # 增加 epoch
    per_device_train_batch_size=8,
    learning_rate=1e-5,  # 降低學習率
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,  # 每 10 步記錄一次
    evaluation_strategy="epoch"
)

# 初始化 Trainer
trainer = Trainer(
    model=model,                      # 模型
    args=training_args,               # 訓練參數
    train_dataset=dataset,            # 訓練資料集
    eval_dataset=dataset              # 評估資料集（用同一批資料）
)

# 開始訓練
trainer.train()

# 保存模型和 tokenizer
model.save_pretrained("./my-DistilBert-classifier")
tokenizer.save_pretrained("./my-DistilBert-classifier")

print("模型和 tokenizer 已成功保存！")
