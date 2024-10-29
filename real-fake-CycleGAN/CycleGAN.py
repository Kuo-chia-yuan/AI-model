import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import itertools

# 確認是否有 GPU 可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, stride=1, padding=1)
        )

    def forward(self, x):
        return torch.sigmoid(self.model(x))

def load_data(data_path, image_size=256, batch_size=1):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 標準化到 [-1, 1]
    ])
    dataset = datasets.ImageFolder(data_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 載入 trainA 和 trainB 中的圖片
trainA_loader = load_data('D:/Jalen/AI_model/style-transfer/my_cyclegan_dataset/real')
trainB_loader = load_data('D:/Jalen/AI_model/style-transfer/my_cyclegan_dataset/fake')

# 初始化生成器和判別器，並傳到 GPU（若可用）
G_A = Generator().to(device)
G_B = Generator().to(device)
D_A = Discriminator().to(device)
D_B = Discriminator().to(device)

# 損失函數
adversarial_loss = torch.nn.MSELoss().to(device)  # 將損失函數也搬到 GPU
cycle_loss = torch.nn.L1Loss().to(device)

# 優化器
optimizer_G = torch.optim.Adam(itertools.chain(G_A.parameters(), G_B.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 訓練循環
for epoch in range(2):
    total_loss_G = 0.0
    total_loss_D_A = 0.0
    total_loss_D_B = 0.0

    for i, (real_A, real_B) in enumerate(zip(trainA_loader, trainB_loader)):
        # 將圖片傳到 GPU（若可用）
        real_A = real_A[0].to(device)
        real_B = real_B[0].to(device)

        # 更新生成器 G_A 和 G_B
        optimizer_G.zero_grad()
        fake_B = G_A(real_A)
        reconstructed_A = G_B(fake_B)
        fake_A = G_B(real_B)
        reconstructed_B = G_A(fake_A)

        # 計算生成器的損失
        loss_G_A2B = adversarial_loss(D_B(fake_B), torch.ones_like(D_B(fake_B)).to(device))
        loss_G_B2A = adversarial_loss(D_A(fake_A), torch.ones_like(D_A(fake_A)).to(device))
        loss_cycle_A = cycle_loss(reconstructed_A, real_A)
        loss_cycle_B = cycle_loss(reconstructed_B, real_B)
        loss_G = loss_G_A2B + loss_G_B2A + 10.0 * (loss_cycle_A + loss_cycle_B)
        loss_G.backward()
        optimizer_G.step()

        # 更新判別器 D_A 和 D_B
        optimizer_D_A.zero_grad()
        loss_D_A = adversarial_loss(D_A(real_A), torch.ones_like(D_A(real_A)).to(device)) + \
                   adversarial_loss(D_A(fake_A.detach()), torch.zeros_like(D_A(fake_A)).to(device))
        loss_D_A.backward()
        optimizer_D_A.step()

        optimizer_D_B.zero_grad()
        loss_D_B = adversarial_loss(D_B(real_B), torch.ones_like(D_B(real_B)).to(device)) + \
                   adversarial_loss(D_B(fake_B.detach()), torch.zeros_like(D_B(fake_B)).to(device))
        loss_D_B.backward()
        optimizer_D_B.step()

        # 累積損失
        total_loss_G += loss_G.item()
        total_loss_D_A += loss_D_A.item()
        total_loss_D_B += loss_D_B.item()

        # 每 100 個 batch 印出一次損失
        if (i + 1) % 100 == 0:
            print(f"[Epoch {epoch+1}/100] [Batch {i+1}] [Loss G: {loss_G.item():.4f}] "
                  f"[Loss D_A: {loss_D_A.item():.4f}] [Loss D_B: {loss_D_B.item():.4f}]")

    # 印出平均損失
    avg_loss_G = total_loss_G / len(trainA_loader)
    avg_loss_D_A = total_loss_D_A / len(trainA_loader)
    avg_loss_D_B = total_loss_D_B / len(trainA_loader)

    print(f"[Epoch {epoch+1}/100] [Avg Loss G: {avg_loss_G:.4f}] "
          f"[Avg Loss D_A: {avg_loss_D_A:.4f}] [Avg Loss D_B: {avg_loss_D_B:.4f}]")

# 儲存模型
torch.save(G_A.state_dict(), "G_A.pth")
torch.save(G_B.state_dict(), "G_B.pth")
torch.save(D_A.state_dict(), "D_A.pth")
torch.save(D_B.state_dict(), "D_B.pth")
print("模型已成功儲存！")
