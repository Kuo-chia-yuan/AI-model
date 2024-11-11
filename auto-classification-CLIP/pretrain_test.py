import os
import clip
import torch
from PIL import Image
import random

# 載入 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 設定圖片與文字描述的路徑
image_folder = "D:/Jalen/AI_model/CLIP/Images"
captions_file = "D:/Jalen/AI_model/CLIP/captions.txt"

# 讀取 captions.txt 檔案
captions_dict = {}
with open(captions_file, 'r') as f:
    next(f)  # 跳過標頭
    for line in f:
        image_file, caption = line.strip().split(',', 1)
        if image_file in captions_dict:
            captions_dict[image_file].append(caption)
        else:
            captions_dict[image_file] = [caption]

# 測試 CLIP 模型
correct = 0
total = 0
batch_size = 1000  # 每 1000 張圖片輸出一次準確率

# 假設我們每張圖片使用5個正確描述和5個隨機干擾描述
num_distractions = 5

for i, (image_file, captions) in enumerate(captions_dict.items(), 1):
    # 載入並處理圖片
    image_path = os.path.join(image_folder, image_file)
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    # 隨機選取其他圖片的描述作為干擾項
    all_captions = list(captions_dict.values())
    distractions = []
    while len(distractions) < num_distractions:
        random_caption = random.choice(random.choice(all_captions))
        if random_caption not in captions:
            distractions.append(random_caption)

    # 將正確描述和干擾描述合併
    test_captions = captions + distractions
    text_tokens = clip.tokenize(test_captions).to(device)
    
    # 計算圖片和文字的相似度
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

        # 計算相似度
        logits_per_image, logits_per_text = model(image, text_tokens)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # 找到最相似的文字
    max_index = probs.argmax()
    best_caption = test_captions[max_index]

    # 判斷預測結果是否在正確描述中
    if best_caption in captions:
        correct += 1
    total += 1

    # 每1000張圖片輸出一次準確率
    if i % batch_size == 0:
        accuracy = correct / total
        print(f"Processed {i} images - Current Accuracy: {accuracy * 100:.2f}%")

# 最終準確率
accuracy = correct / total
print(f"Final Accuracy: {accuracy * 100:.2f}%")