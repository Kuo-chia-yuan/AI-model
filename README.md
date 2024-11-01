## AI model
![image](https://github.com/user-attachments/assets/4d801138-8313-40cb-ba1f-76a34533cb38)

## 利用 loss, lr 更新參數
![image](https://github.com/user-attachments/assets/2453fcfb-82c7-42c3-ba9c-ba5214c97166)

## activation & loss function
![image](https://github.com/user-attachments/assets/f2ff5b95-0b62-4ff4-895f-cfd0ef78615a)

## batch size
![image](https://github.com/user-attachments/assets/ded44e1a-a92a-450f-b30f-ab09d8742ecf)

## Fine-tune
![image](https://github.com/user-attachments/assets/3fb6649e-364c-4218-b216-062a2712687e)

## 自動化 Trainer VS 手刻迴圈訓練
![image](https://github.com/user-attachments/assets/ea2a81f5-c86b-4e87-a13c-6d4e5e436b05)

## Linear regression
![image](https://github.com/user-attachments/assets/cfbb62fa-a469-4411-bc4a-3b198d229649)
* 線性關係：y = w * x + b
* 雖然簡陋，但可避免過擬合

## RNN & BRNN & LSTM
![image](https://github.com/user-attachments/assets/48621381-3b0c-41c2-80d3-a8a4bc9ac83a)
* X_t：input
* h_t：output
* C_t：長期記憶

![image](https://github.com/user-attachments/assets/4f734782-d2c3-4b58-bb26-b3ee49667b81)

### Image_Captioning: LSTM + language model
![image](https://github.com/user-attachments/assets/9e7e1d9f-e549-4176-89fc-529570312d11)

![image](https://github.com/user-attachments/assets/f02a2409-aad4-478b-ace8-2adef02d05ea)

* Dictionary：建立詞與 ID 之間的映射，如 "我" -> 5、"帥" -> 10
* Corpus (語料庫)：將句子轉換成 ID 序列，如 "我 好 帥" 會被轉換成 [5, 2, 10]
* --------- 前向傳播 ---------
* Embedding：建立詞彙表，將 ID 各別轉換成 Embedding 向量，如 ID 5 -> 向量 [0.2, 0.1, 0.3, 0.7]
* LSTM：學習句子中的上下文關係
* Linear：輸出下一個詞的概率分佈，如，輸出 [0.1, 0.2, 0.05, 0.65] 表示對應的四個詞的概率，最大概率的詞是模型的預測詞

### CNN (encoder) + LSTM (decoder)
![image](https://github.com/user-attachments/assets/d40aacb3-cc03-4d14-8dc3-967d2071c163)

## Denoise - DnCNN
![image](https://github.com/user-attachments/assets/e5b6ee92-9f6f-440d-ae53-dfcf5ba6274d)

## Object Detection - YOLO
![image](https://github.com/user-attachments/assets/5c8b194e-2407-4b38-8647-08a04d01c5ed)

![image](https://github.com/user-attachments/assets/67eb5a43-21ce-48d2-b6af-7c1328e64d84)

![image](https://github.com/user-attachments/assets/4fe2bddc-d772-4554-80e3-213fe9c0faf5)

## ResNet
![image](https://github.com/user-attachments/assets/12c68d44-9d10-4a35-aa6f-ff8b3cf6ee47)

## GAN
![image](https://github.com/user-attachments/assets/a9ced60c-32cf-466c-a771-68d256854ca4)

![image](https://github.com/user-attachments/assets/c9f03e72-6869-4eb3-bf48-2b8d10f9b3d3)

## Neural Style Transfer - VGG19
![image](https://github.com/user-attachments/assets/88e4c9b6-fc28-4fe3-b890-4d4d76371077)

![image](https://github.com/user-attachments/assets/0f7f7a00-3a9f-4fdb-b920-0c8368eb5f5e)

![image](https://github.com/user-attachments/assets/4928906b-3f56-4c79-a90a-3057c0b2ffb9)

## StyleGAN
![image](https://github.com/user-attachments/assets/086fa6e4-2552-45b5-a070-80570c036122)

![image](https://github.com/user-attachments/assets/5d54c475-27fa-4ff2-bceb-483cfe91d113)

## CycleGAN
![image](https://github.com/user-attachments/assets/30e970bc-3f29-43d1-95a7-fb9973abf52d)

* VGG19：input 是 content image、style image、noise image / loss 是 noise image 與 content image、style image 的差異
* StyleGAN：input 是 noise image / loss 是 G_loss 和 D_loss
* CycleGAN：input 是 real image、fake image / loss 是 G_real_loss、G_fake_loss、D_real_loss、D_fake_loss、real_to_fake_to_real、fale_to_real_to_fake

## VAE (Variational Autoencoder)
![image](https://github.com/user-attachments/assets/7801a9dc-1853-4df1-b482-10c9dd5b6251)

![image](https://github.com/user-attachments/assets/faba85e3-a770-40f9-b068-3f643b51007e)

* m: 由 fc2 求出的均值
* σ: 由 fc3 求出的標準差
* e: 0 到 1 之間的高斯分布
* loss_1: output 與 input 的差異
* loss_2: output 與高斯分布的差異 (為了滿足生成的多樣性)

## Segmentation - UNet
![image](https://github.com/user-attachments/assets/41625379-8369-4620-b4c0-a38869747720)

## Seq2Seq
![image](https://github.com/user-attachments/assets/6383b2fd-db18-4e5d-a30d-f91337765edc)

![image](https://github.com/user-attachments/assets/2006762b-e0ae-431d-8d25-8ee7a19d9496)
* Seq2Seq 加入 Attention 後，Decoder input 的 Query 向量會與每個 Encoder output 的 key 和 value 向量相乘並加權求和後，輸出 output，類似 Transformer 的 Cross Attention

## Transformer
![image](https://github.com/user-attachments/assets/91aa5f29-4ffa-435e-9e59-6ac08504634c)

![image](https://github.com/user-attachments/assets/89d4e7c5-0f4e-4c62-aaae-a48b44441bde)

![image](https://github.com/user-attachments/assets/ac0ff115-7f53-469e-b16c-40bc225b8c41)

![image](https://github.com/user-attachments/assets/2d0ab570-46e5-4640-9300-c7b101e00bbb)

![image](https://github.com/user-attachments/assets/b7cdab04-9c44-48f4-9a33-99c1d8e0d20d)
* 近期 Seq2Seq 已用 Transformer 取代 LSTM 

### BERT
![image](https://github.com/user-attachments/assets/80734c3a-9c66-4e2f-92ea-3d242b9df4c8)

![image](https://github.com/user-attachments/assets/53c6a80b-bcaf-409a-889d-3568dff4a00e)

* loss_1: Masked Language Model (MLM), 隨機選擇一些詞語（token）進行「遮蔽」（mask），為分類問題，每個被遮蔽的 token 都需要從詞彙表（vocabulary）中預測出正確的詞
* loss_2: Next Sentence Prediction (NSP), 給模型一對句子（A 和 B），預測 B 是否是 A 的正確後續句

### GPT
![image](https://github.com/user-attachments/assets/414ef9f9-1bbd-4581-9cab-aad051e770fa)

![image](https://github.com/user-attachments/assets/4d83d095-b69d-4286-b5cb-48db1bc33daa)
