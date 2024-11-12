import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim import Adam
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# 自定義單一文件夾影像的 Dataset
class ImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = "train"
        self.transform = transform
        self.image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")  # 確保轉換為 RGB 格式

        if self.transform:
            image = self.transform(image)

        return image, 0  # 因為沒有標籤，這裡可以直接返回 0 作為佔位標籤

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # 初始化嵌入向量表
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, inputs):
        # 將輸入展平
        flat_input = inputs.view(-1, self.embedding_dim)

        # 計算每個輸入向量與嵌入向量的距離
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embeddings.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))

        # 找到最近的嵌入向量的索引
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = self.embeddings(encoding_indices).view(inputs.shape)

        # 計算損失：commitment loss 和 quantization loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # 將量化後的向量和梯度結合
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, encoding_indices

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, embedding_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)  # 批量正規化

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim * 2)  # 批量正規化

        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_dim * 4)  # 批量正規化

        self.conv4 = nn.Conv2d(hidden_dim * 4, embedding_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x


class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(embedding_dim, hidden_dim * 4, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim * 4)  # 批量正規化

        self.conv2 = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim * 2)  # 批量正規化

        self.conv3 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_dim)  # 批量正規化

        self.conv4 = nn.ConvTranspose2d(hidden_dim, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.sigmoid(self.conv4(x))  # 使用 sigmoid 確保輸出範圍在 [0, 1]
        return x

class VQVAE(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=128, embedding_dim=64, num_embeddings=512, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(in_channels, hidden_dim, embedding_dim)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, hidden_dim, in_channels)

    def forward(self, x):
        # 編碼
        z = self.encoder(x)
        # 向量量化
        quantized, vq_loss, _ = self.vq_layer(z)
        # 解碼
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss


if __name__ == "__main__":
    
    # 定義影像轉換（包括縮放和歸一化等）
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 調整影像大小為 32x32
        transforms.ToTensor(),        # 轉換為 Tensor 並將值縮放到 [0, 1]
    ])

    # 建立 Dataset 和 DataLoader
    train_dataset = ImageDataset(folder_path="train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

    # 移動模型到 GPU（如果有的話）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQVAE().to(device)

    # 設定優化器和訓練參數
    optimizer = Adam(model.parameters(), lr=1e-3)
    num_epochs = 100

    # 開始訓練
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)

            # 前向傳播
            recon_images, vq_loss = model(images)
            recon_loss = F.mse_loss(recon_images, images)

            # 總損失
            loss = recon_loss + vq_loss

            # 反向傳播和參數更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累積損失以便後續監控
            running_loss += loss.item()

        # 每個 epoch 結束後輸出平均損失
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Recon Loss: {recon_loss.item():.4f}, VQ Loss: {vq_loss.item():.4f}")

        # 每隔幾個 epoch 可視化重建結果
        if (epoch + 1)==100:
            model.eval()
            with torch.no_grad():
                for images, _ in train_loader:
                    images = images.to(device)
                    recon_images, _ = model(images)
                    
                    # 顯示原圖和重建圖
                    fig, axs = plt.subplots(1, 2)
                    axs[0].imshow(images[0].permute(1, 2, 0).cpu().numpy())
                    axs[0].set_title("Original")
                    axs[0].axis("off")

                    axs[1].imshow(recon_images[0].permute(1, 2, 0).cpu().numpy())
                    axs[1].set_title("Reconstructed")
                    axs[1].axis("off")

                    plt.show()
                    break  # 只顯示一個 batch
            model.train()  # 恢復到訓練模式