import pandas as pd
import os
os.environ["WANDB_DISABLED"] = "true"

# 載入中文與台語的 CSV 檔案
zh_data = pd.read_csv("train-ZH.csv")  # 中文句子
tl_data = pd.read_csv("train-TL.csv")  # 台語句子

# 檢查是否筆數對齊
assert len(zh_data) == len(tl_data), "中文與台語資料筆數不一致！"

# 將 txt 欄位轉成 list，準備進行訓練
zh_sentences = zh_data["txt"].tolist()
tl_sentences = tl_data["txt"].tolist()

from transformers import MarianMTModel, MarianTokenizer

# 選擇一個適合的預訓練翻譯模型（可以從其他語言對微調）
model_name = "Helsinki-NLP/opus-mt-zh-en"  # 中翻英模型，可微調成台語
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# 一次性處理輸入（中文句子）和目標（台語句子）
model_inputs = tokenizer(
    zh_sentences,
    text_target=tl_sentences,  # 指定目標句子
    padding=True,
    truncation=True,
    return_tensors="pt"
)

from torch.utils.data import Dataset

# 自定義 Dataset 類，將 tokenized 資料包裝成 Dataset 格式
class TranslationDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: tensor[idx] for key, tensor in self.encodings.items()}
    
# 將處理後的輸入資料轉為 Dataset 格式
train_dataset = TranslationDataset(model_inputs)

from transformers import Seq2SeqTrainingArguments

# 設定訓練參數
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",           # 模型輸出目錄
    eval_strategy="epoch",      # 每個 epoch 進行一次評估
    learning_rate=2e-5,               # 學習率
    per_device_train_batch_size=16,   # 每台設備的訓練批次大小
    per_device_eval_batch_size=16,    # 每台設備的評估批次大小
    num_train_epochs=1,               # 訓練 3 個 epoch
    weight_decay=0.01,                # 權重衰減
    save_total_limit=2,               # 保存最多 2 個模型版本
    logging_dir="./logs",             # 日誌文件目錄
    logging_steps=10,                 # 每 10 步記錄一次
)

from transformers import Seq2SeqTrainer

# 建立 Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model=model,                  # 翻譯模型
    args=training_args,           # 訓練參數
    train_dataset=train_dataset,   # 訓練資料集
    eval_dataset=train_dataset     # 評估資料集（這裡用同一批資料，正式使用時建議分開）
)

# 開始訓練
trainer.train()

# 保存模型和 tokenizer 到指定目錄
model.save_pretrained("./my-tl-translator")
tokenizer.save_pretrained("./my-tl-translator")

print("模型和 tokenizer 已成功保存！")