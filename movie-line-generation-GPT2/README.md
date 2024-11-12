# movie-line-generation-GPT2
## input
- 逐個 token 的 embedding 向量 (768 維)

## embedding
- Token Embedding：將每個 token 映射到對應的 768 維向量
- Positional Encoding：將每個 token 的位置應設到對應的 768 維向量  
(共兩個嵌入層矩陣，無 Segment Embedding)

## embedding pretrain 
- 與 BERT 方式相同

## decoder
- 因為缺少 encoder，所以無 cross attention
![alt text](decoder.png)

## pretrain
採用 Autoregressive Modeling (自回歸模型)  
- ground truth = "我好帥"
    1. 第一次 input = "我"  
    第一次 output = "?"  
    第一次 loss = "?" 和 "好" 的 Cross-Entropy Loss
    2. 第二次 input = "我好"  
    第二次 output = "?"  
    第二次 loss = "?" 和 "帥" 的 Cross-Entropy Loss
    3. 最終 loss = 所有 loss 相加

## fine tune
- 與 pretrain 方式相同

## demo 
![image](https://github.com/user-attachments/assets/552f4ad8-792e-4803-a662-310e712d5a91)

# DALL-E
input：一句話  
output：一張 image  

處理文字：GPT  
處理 image：VQ-VAE (如下圖) + GPT + VQ-VAE    

![image](https://github.com/user-attachments/assets/e080c1ea-1375-4fa7-b97f-e0b4df8b53b4)  

- VQ-VAE：將 image 輸入至 CNN 提取 feature map，並根據 embedding space 將連續特徵轉換為離散 token
- GPT：將 token 通過 embedding 轉成向量，並逐個輸出向量
- VQ-VAE：將每個向量逐個 up sampling 成一小塊 image，最終合併成完整 image
