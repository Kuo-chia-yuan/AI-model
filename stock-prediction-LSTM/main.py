# import yfinance as yf

# # 下載台積電 (TSM) 的歷史資料
# data = yf.download("0050.TW", start="2010-01-01", end="2023-01-01")

# # 將資料儲存為 CSV 檔案
# data.to_csv('0050_history.csv')
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

import pandas as pd

# 將第一欄 (日期) 當作索引
df = pd.read_csv('0050_history.csv', index_col=0, parse_dates=True)

# 選擇 'Adj Close' 作為我們的預測目標
prices = df[['Adj Close']].values  # 將資料轉為 NumPy 陣列

# 使用 MinMaxScaler 進行正規化（縮放至 0~1 之間）
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(prices)

def create_dataset(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# 將資料集拆分為訓練集 (80%) 和測試集 (20%)
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# 創建訓練和測試資料
X_train, y_train = create_dataset(train_data, sequence_length=100)
X_test, y_test = create_dataset(test_data, sequence_length=100)

# 將資料轉為 PyTorch 的 Tensor 格式
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 層
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 全連接層（輸出層）
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化 LSTM 隱藏狀態和記憶單元
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()

        # 前向傳播 LSTM
        out, _ = self.lstm(x, (h0, c0))

        # 取最後一個時間步的輸出，並通過全連接層
        out = self.fc(out[:, -1, :])  # 注意這裡是取最後一個時間步的輸出
        return out

# 初始化模型、損失函數和優化器
model = LSTM()
criterion = nn.MSELoss()  # 使用均方誤差 (MSE) 作為損失函數
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 訓練模型
num_epochs = 100  # 訓練週期數
for epoch in range(num_epochs):
    model.train()  # 設定模型為訓練模式
    outputs = model(X_train)  # 前向傳播
    optimizer.zero_grad()  # 清空梯度

    # 計算損失
    loss = criterion(outputs, y_train)
    loss.backward()  # 反向傳播
    optimizer.step()  # 更新參數

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 儲存模型的權重
torch.save(model.state_dict(), 'lstm_model_weights.pth')
model.eval()  # 設定模型為評估模式
print("模型權重已儲存為 lstm_model_weights.pth")

# 評估模型
with torch.no_grad():
    predictions = model(X_test)  # 預測測試資料

# 將預測結果和真實值還原為原始尺度
predictions = scaler.inverse_transform(predictions.numpy())
y_test = scaler.inverse_transform(y_test.numpy())

# 繪製真實值與預測值的比較圖
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(y_test, label='True Price', color='blue')
plt.plot(predictions, label='Predicted Price', color='red')
plt.title('0050.TW Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()
