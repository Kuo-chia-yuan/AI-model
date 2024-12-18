# projection-classification-ResNet18
***註：此為與某公司合作之 Matlab case***
## target
- 利用 ai 實作影像分類，並與傳統方法進行比較
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
![result1](https://github.com/user-attachments/assets/92fb836f-f9bd-488e-85b2-6b32492f4618)

- 訓練時長：1911 hr
- 訓練 epoch：10
- validation acc：96.37
- validation loss：0.1
