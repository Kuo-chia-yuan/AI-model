import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.utils import save_image
from tqdm import tqdm

# 1. 生成器 (Generator) 定義
class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels, img_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, img_channels * img_size * img_size),
            nn.Tanh()
        )
        self.img_size = img_size
        self.img_channels = img_channels

    def forward(self, z):
        img = self.model(z)
        return img.view(z.size(0), self.img_channels, self.img_size, self.img_size)

# 2. 鑑別器 (Discriminator) 定義
class Discriminator(nn.Module):
    def __init__(self, img_channels, img_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_channels * img_size * img_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# 3. 設定超參數
latent_dim = 100
img_size = 64
img_channels = 3  # 彩色影像 (RGB)
batch_size = 32
learning_rate = 0.0002
epochs = 1000

# 4. 自定義資料集處理
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset = datasets.ImageFolder(root='D:/Jalen/AI_model/style-transfer/dataset/real',
                               transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 5. 初始化生成器和鑑別器
generator = Generator(latent_dim, img_channels, img_size).cuda()
discriminator = Discriminator(img_channels, img_size).cuda()

# 6. 損失函數與優化器
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# 7. 訓練迴圈
os.makedirs('images_result2', exist_ok=True)  # 儲存生成影像的資料夾

for epoch in range(epochs):
    for i, (imgs, _) in enumerate(tqdm(dataloader)):
        real_imgs = imgs.cuda()
        batch_size = real_imgs.size(0)

        # 真實影像的標籤為 1，生成影像的標籤為 0
        valid = torch.ones((batch_size, 1)).cuda()
        fake = torch.zeros((batch_size, 1)).cuda()

        # -----------------
        #  訓練生成器 (G)
        # -----------------
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_dim).cuda()  # 隨機噪聲
        generated_imgs = generator(z)  # 生成影像
        g_loss = adversarial_loss(discriminator(generated_imgs), valid)  # 希望生成影像被判為真
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  訓練鑑別器 (D)
        # ---------------------
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs), valid)  # 真實影像的損失
        fake_loss = adversarial_loss(discriminator(generated_imgs.detach()), fake)  # 生成影像的損失
        d_loss = (real_loss + fake_loss) / 2  # 總損失
        d_loss.backward()
        optimizer_D.step()

        # 每 100 個批次儲存一次生成影像
        if i % 100 == 0:
            save_image(generated_imgs.data[:25], f"images_result2/{epoch}_{i}.png", nrow=5, normalize=True)

    print(f"[Epoch {epoch}/{epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

# 8. 儲存模型
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")

print("保存完畢")
