# projection-classification-ResNet18
***註：此為與某公司合作之 Matlab case***
## target
- 主要負責利用 ai 實作影像分類，並與傳統方法進行比較
## input
- 由優至劣，分為 A、B、C、D、E 等級
- 每個等級各 12 萬張 projection data

## output
- 判斷一張 projection data 是哪個等級

## 超參數設定
- MaxEpoch = 10
- MinBatch = 32
- Initial LR = 0.0005

## result
![result](https://github.com/user-attachments/assets/c87e6161-7b85-40df-9919-ee4ab334effc)
- 訓練時長：2 天
- 訓練 epoch：1.2
- acc：training data 90 % / validation data 80 %
- loss：0.3
