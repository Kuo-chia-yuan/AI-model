# character-classification-ResNet18
## data augmentation
- Resize：調整尺寸
- RandomRotation：隨機旋轉
- RandomHorizontalFlip：隨機水平翻轉
- add_random_noise：加入雜訊
- Pad：填充黑邊

## input
- 224 x 224 x 3 image

## output
- 50 種類別的各自概率，最大為 1，其餘為 0

## architecture
![image](https://github.com/user-attachments/assets/e32b53e4-f868-4e28-8481-698490da43ea)
- 殘差塊 (residual block)：(CNN + Batch Normalization + ReLU) + (CNN + Batch Normalization)

## loss
- Cross-Entropy Loss (CNN)  
![alt text](loss.png)
- softmax (最後一層輸出)

## key
- 殘差 (residual) = x + f(x-1)
