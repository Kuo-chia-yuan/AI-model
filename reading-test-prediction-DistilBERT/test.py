import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# 1. 載入已保存的模型和 tokenizer
model = DistilBertForSequenceClassification.from_pretrained("./my-DistilBert-classifier")
tokenizer = DistilBertTokenizer.from_pretrained("./my-DistilBert-classifier")

# 設定模型為評估模式
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 2. 載入 AI1000.xlsx 並檢查資料格式
df_new = pd.read_excel('AI1000.xlsx')  # 確保 '題號' 和 '文章' 欄位存在

# 3. 合併「文章」欄位成輸入特徵（與訓練格式一致）
df_new['輸入特徵'] = df_new['文章'].astype(str)

# 定義 batch size，避免一次性處理過多資料
BATCH_SIZE = 8  # 根據 GPU 記憶體調整

# 4. Tokenize 並分批處理資料
def encode_texts(texts):
    return tokenizer(
        texts,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

all_answers = []  # 儲存所有預測結果

# 分批處理資料避免 CUDA Out of Memory
for i in range(0, len(df_new), BATCH_SIZE):
    # 取得該批次資料
    batch_texts = df_new['輸入特徵'].tolist()[i:i+BATCH_SIZE]
    X_encodings = encode_texts(batch_texts)

    with torch.no_grad():
        # 將張量移至 GPU 或 CPU
        input_ids = X_encodings['input_ids'].to(device)
        attention_mask = X_encodings['attention_mask'].to(device)

        # 進行模型推論，取得 logits
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # 找到最高分的類別（對應於 1~4 選項）
        predictions = torch.argmax(logits, dim=1)  # 索引為 0~3
        answers = predictions.cpu().numpy() + 1  # 調整回 1~4 範圍

        # 將該批次的答案加入到 all_answers 中
        all_answers.extend(answers)

# 5. 將結果保存為 Excel 檔案
df_result = pd.DataFrame({
    'ID': df_new['題號'],  # 題號欄位與原始資料對應
    'Answer': all_answers  # 預測的答案
})

# 保存結果到 AI1000_predictions.xlsx
df_result.to_excel('AI1000_predictions.xlsx', index=False)

print("預測完成，結果已保存至 AI1000_predictions.xlsx！")
