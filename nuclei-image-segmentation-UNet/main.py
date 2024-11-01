import os
from glob import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class UNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, image_size=(512, 512)):
        """
        初始化數據集
        :param root_dir: 根目錄，包含所有樣本的子資料夾
        :param transform: 影像的轉換 (augmentation)
        :param image_size: 影像尺寸，用於resize
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        self.samples = sorted(os.listdir(root_dir))  # 每個樣本子資料夾的名稱

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = os.path.join(self.root_dir, self.samples[idx])
        
        # 讀取影像
        image_path = glob(os.path.join(sample_dir, 'images', '*.png'))[0]
        image = Image.open(image_path).convert('RGB')  # 假設影像是RGB格式
        
        # 讀取並合併所有的mask
        mask_paths = glob(os.path.join(sample_dir, 'masks', '*.png'))
        combined_mask = np.zeros(self.image_size, dtype=np.uint8)
        for mask_path in mask_paths:
            mask = Image.open(mask_path).convert('L')  # 轉為灰度圖
            mask = mask.resize(self.image_size)  # 調整大小
            mask = np.array(mask)
            combined_mask = np.maximum(combined_mask, mask)  # 合併多個mask
        
        # 轉換為Tensor
        if self.transform:
            image = self.transform(image)
        else:
            # 基本轉換
            transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ])
            image = transform(image)
        
        # mask轉為Tensor並標準化
        combined_mask = Image.fromarray(combined_mask)
        combined_mask = transforms.ToTensor()(combined_mask)  # 轉換為Tensor
        combined_mask = (combined_mask > 0).float()  # 將mask的值限制為0或1
        
        return image, combined_mask

# 設置路徑和參數
root_dir = 'D:/Jalen/AI_model/UNet/training-data'
image_size = (512, 512)  # 調整尺寸，根據需求更改

# 初始化數據集
dataset = UNetDataset(root_dir=root_dir, image_size=image_size)

# 建立 DataLoader
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 測試讀取數據
for images, masks in data_loader:
    print("Images batch shape:", images.shape)  # (batch_size, 3, H, W)
    print("Masks batch shape:", masks.shape)  # (batch_size, 1, H, W)
    break

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)
        self.center = conv_block(512, 1024)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)
        
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        e3 = self.encoder3(F.max_pool2d(e2, 2))
        e4 = self.encoder4(F.max_pool2d(e3, 2))
        
        center = self.center(F.max_pool2d(e4, 2))
        
        d4 = self.decoder4(torch.cat([self.up4(center), e4], 1))
        d3 = self.decoder3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.decoder2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.decoder1(torch.cat([self.up1(d2), e1], 1))
        
        return torch.sigmoid(self.final(d1))

def main():
    # 建立模型
    model = UNet(in_channels=3, out_channels=1)

    import torch.optim as optim

    # 損失函數
    criterion = nn.BCELoss()  # 二元交叉熵損失

    # 優化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, masks) in enumerate(data_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            # 前向傳播
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # 反向傳播和優化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(data_loader)}], Loss: {loss.item():.4f}")

        # 每個 epoch 結束時，計算並輸出平均損失
        epoch_loss = running_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), "unet_model.pth")
    print("Model saved.")

if __name__ == "__main__":
    main()