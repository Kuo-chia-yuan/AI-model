# reading-test-prediction-DistilBERT
## input
- [CLS] + 第一段 + [SEP] + 第二段 + [SEP] + 第三段 + [SEP]

## embedding
- 最終輸入向量 = Token Embedding + Position Embedding (兩者皆是 768 維向量)

## embedding training
- 直接使用 BERT pretrain 完成的嵌入層矩陣 (Token Embedding 及 Position Embedding)，來初始化 DistilBERT 的嵌入層矩陣

## encoder (DistilBERT 只有 6 層 encoder)
- 與 BERT encoder 架構相同

## pretrain
- 去除 NSP：因為缺少 Segment Embedding，無法找出相連的 AB 兩句
- 保留 MLM，同時使用知識蒸餾 (Knowledge Distillation)：
    1. 獲得已完成預訓練 (MLM + NSP) 的 BERT
    2. 用 training data 訓練 DistilBERT，並預測 [MASK] 向量
    3. 用相同 training data 執行 BERT，並得到 [MASK] 向量
    4. loss = (DistilBERT 與 BERT 輸出的差異) + (DistilBERT 與正確答案的差異)

## fine tune
- 與 BERT 相同