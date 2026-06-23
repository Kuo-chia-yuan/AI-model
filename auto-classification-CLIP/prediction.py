import os
import clip
import torch
from PIL import Image
import random
import matplotlib.pyplot as plt

# 載入 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 設定圖片資料夾路徑與選取圖片
# （備註：您程式碼中雖然隨機挑選了 random_image_file，但實際讀取路徑是寫死指定的圖片）
image_path = "/content/yu-kato-cddaZDt6uRw-unsplash.jpg"
raw_image = Image.open(image_path)

# 定義零樣本分類的類別名稱
categories = ["animal", "man", "woman", "child", "landscape"]
category_descriptions = [f"A photo of a {category}" for category in categories]

# 將類別名稱轉換成 CLIP 模型的文字向量
text_tokens = clip.tokenize(category_descriptions).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)

# 載入並處理圖片
image = preprocess(raw_image).unsqueeze(0).to(device)

# 將圖片編碼為向量並計算相似度
with torch.no_grad():
    image_features = model.encode_image(image)
    logits_per_image, logits_per_text = model(image, text_tokens)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# 找出最相似的類別名稱
max_index = probs.argmax()
predicted_category = categories[max_index]
prob_percentages = probs[0] * 100

# ================= 視覺化 Output 區塊 =================

# 1. 建立一個 1 列 2 行的畫布 (左邊放圖，右邊放圖表)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 2. 左圖：顯示原始圖片
ax1.imshow(raw_image)
ax1.set_title("Input Image", fontsize=14, fontweight='bold')
ax1.axis('off')  # 隱藏圖片的座標軸

# 3. 右圖：繪製水平長條圖
# 讓預測成功的長條顯示為橘色 (orange)，其餘為天藍色 (skyblue)
colors = ['orange' if cat == predicted_category else 'skyblue' for cat in categories]
bars = ax2.barh(categories, prob_percentages, color=colors, edgecolor='black')

# 4. 在每個長條圖後方標註百分比數字
for bar in bars:
    width = bar.get_width()
    ax2.text(width + 1, 
             bar.get_y() + bar.get_height()/2, 
             f'{width:.2f}%', 
             ha='left', va='center', fontsize=11, fontweight='bold')

# 5. 設定圖表細節
ax2.set_title(f"Prediction: {predicted_category}", fontsize=14, fontweight='bold', color='darkorange')
ax2.set_xlabel("Probability (%)", fontsize=12)
ax2.set_xlim(0, 115)  # 留空間給標籤文字
ax2.invert_yaxis()    # 讓第一個類別在最上方
ax2.grid(axis='x', linestyle='--', alpha=0.5)

# 顯示最終完美的並排結果
plt.tight_layout()
plt.show()
