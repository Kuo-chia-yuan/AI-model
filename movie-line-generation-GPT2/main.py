from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

# 加載 GPT-2 模型和 tokenizer
model_name = "gpt2"  # 可換成 "gpt2-medium" 或 "gpt2-large" 看需求和硬體資源
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 加載並準備訓練資料
train_path = "training_data.txt"  # 你的訓練資料文件路徑

# 定義文本資料集
def load_dataset(file_path):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128  # 每段文本的 token 長度，可根據需要調整
    )

# 資料加載器（自動填充 batch 中的資料）
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT 是自回歸模型，非 Masked Language Model，因此 mlm=False
)

train_dataset = load_dataset(train_path)

# 訓練參數
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",  # 模型輸出的保存路徑
    overwrite_output_dir=True,
    num_train_epochs=3,  # 訓練的 epoch 數
    per_device_train_batch_size=4,  # 根據 GPU 記憶體大小調整
    save_steps=500,  # 每隔多少步儲存模型
    save_total_limit=2,  # 最多保留幾個模型檔案
    logging_dir="./logs",  # 訓練日誌路徑
)

# 使用 Trainer 進行訓練
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# 開始訓練
trainer.train()

# 保存模型
trainer.save_model("./gpt2-finetuned")
