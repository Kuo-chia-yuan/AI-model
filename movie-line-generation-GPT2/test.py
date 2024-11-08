from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

# 從 Hugging Face 的 GPT-2 預訓練模型中加載 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 加載微調過的 GPT-2 模型和 tokenizer
model_path = "./gpt2-finetuned/checkpoint-27114"  # 你的模型保存路徑
model = GPT2LMHeadModel.from_pretrained(model_path)

# 定義輸入起始文本（prompt）
input_text = "How the weather today?"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# 使用模型生成句子
output = model.generate(
    input_ids,
    max_length=50,           # 設定生成的最大長度
    num_return_sequences=1,   # 生成的句子數量
    no_repeat_ngram_size=2,   # 避免重複生成相同的 n-gram
    top_k=50,                 # 只考慮概率最高的 50 個 token
    top_p=0.95,               # 使用 nucleus sampling 篩選 token
    temperature=0.7           # 控制生成文本的隨機性，較低值生成更保守的文本
)

# 將生成的 token 轉換為句子
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
