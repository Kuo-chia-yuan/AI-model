import os
import clip
import torch
from PIL import Image
import random

# 載入 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 設定圖片資料夾路徑
image_folder = "D:/Jalen/AI_model/CLIP/Images"

# 定義零樣本分類的類別名稱
categories = ["animal", "man", "woman", "child", "landscape"]
category_descriptions = [f"A photo of a {category}" for category in categories]

# 將類別名稱轉換成 CLIP 模型的文字向量
text_tokens = clip.tokenize(category_descriptions).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)

# 隨機選擇一張圖片
image_files = os.listdir(image_folder)
random_image_file = random.choice(image_files)
image_path = "D:/Jalen/AI_model/CLIP/Images/108899015_bf36131a57.jpg"

# 載入並處理圖片
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

# 將圖片編碼為向量
with torch.no_grad():
    image_features = model.encode_image(image)

    # 計算相似度
    logits_per_image, logits_per_text = model(image, text_tokens)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# 找出最相似的類別名稱
max_index = probs.argmax()
predicted_category = categories[max_index]

# 輸出結果
print(f"Randomly selected image: {random_image_file}")
print(f"Predicted category: {predicted_category}")
print("Category probabilities:")
for i, category in enumerate(categories):
    print(f"{category}: {probs[0][i] * 100:.2f}%")
