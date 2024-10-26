import pandas as pd
from transformers import MarianMTModel, MarianTokenizer

# 1. 載入已保存的模型和 tokenizer
model = MarianMTModel.from_pretrained("./my-tl-translator")
tokenizer = MarianTokenizer.from_pretrained("./my-tl-translator")

print("模型和 tokenizer 已成功載入！")

# 2. 讀取 test-ZH-nospace.csv 檔案
test_data = pd.read_csv("test-ZH-nospace.csv")

# 3. 提取中文句子
zh_sentences = test_data["txt"].tolist()

# 4. 將中文句子編碼為模型可接受的格式
inputs = tokenizer(zh_sentences, return_tensors="pt", padding=True, truncation=True)

# 5. 使用模型生成台語翻譯
print("正在進行翻譯...")
translated_tokens = model.generate(**inputs)

# 6. 將生成的 token 解碼為台語句子
tl_sentences = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]

# 7. 將結果保存為新的 DataFrame
output_data = pd.DataFrame({"id": test_data["id"], "txt": tl_sentences})

# 8. 保存為 CSV 檔案
output_data.to_csv("test-TL.csv", index=False, encoding="utf-8-sig")

print("翻譯完成並保存為 test-TL.csv！")
