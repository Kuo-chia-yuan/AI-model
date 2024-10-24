## AI model
![image](https://github.com/user-attachments/assets/4d801138-8313-40cb-ba1f-76a34533cb38)

## 利用 loss, lr 更新參數
![image](https://github.com/user-attachments/assets/2453fcfb-82c7-42c3-ba9c-ba5214c97166)

## activation & loss function
![image](https://github.com/user-attachments/assets/f2ff5b95-0b62-4ff4-895f-cfd0ef78615a)

## batch size
![image](https://github.com/user-attachments/assets/ded44e1a-a92a-450f-b30f-ab09d8742ecf)

## Linear regression
![image](https://github.com/user-attachments/assets/cfbb62fa-a469-4411-bc4a-3b198d229649)
* 線性關係：y = w * x + b
* 雖然簡陋，但可避免過擬合

## RNN & BRNN & LSTM
![image](https://github.com/user-attachments/assets/48621381-3b0c-41c2-80d3-a8a4bc9ac83a)

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

## ResNet
![image](https://github.com/user-attachments/assets/12c68d44-9d10-4a35-aa6f-ff8b3cf6ee47)

## GAN
![image](https://github.com/user-attachments/assets/a9ced60c-32cf-466c-a771-68d256854ca4)

![image](https://github.com/user-attachments/assets/c9f03e72-6869-4eb3-bf48-2b8d10f9b3d3)

## Neural Style Transfer - VGG19
![image](https://github.com/user-attachments/assets/88e4c9b6-fc28-4fe3-b890-4d4d76371077)

![image](https://github.com/user-attachments/assets/0f7f7a00-3a9f-4fdb-b920-0c8368eb5f5e)

![image](https://github.com/user-attachments/assets/4928906b-3f56-4c79-a90a-3057c0b2ffb9)

## styleGAN
![image](https://github.com/user-attachments/assets/086fa6e4-2552-45b5-a070-80570c036122)

![image](https://github.com/user-attachments/assets/5d54c475-27fa-4ff2-bceb-483cfe91d113)

## VAE
![image](https://github.com/user-attachments/assets/7801a9dc-1853-4df1-b482-10c9dd5b6251)

![image](https://github.com/user-attachments/assets/faba85e3-a770-40f9-b068-3f643b51007e)

* m: 由 fc2 求出的均值
* σ: 由 fc3 求出的標準差
* e: 0 到 1 之間的高斯分布
* loss_1: output 與 input 的差異
* loss_2: output 與高斯分布的差異 (為了滿足生成的多樣性)

## U-Net
![image](https://github.com/user-attachments/assets/41625379-8369-4620-b4c0-a38869747720)

## Transformer
![image](https://github.com/user-attachments/assets/91aa5f29-4ffa-435e-9e59-6ac08504634c)

![image](https://github.com/user-attachments/assets/89d4e7c5-0f4e-4c62-aaae-a48b44441bde)

![image](https://github.com/user-attachments/assets/ac0ff115-7f53-469e-b16c-40bc225b8c41)
