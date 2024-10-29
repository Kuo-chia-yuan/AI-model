import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# 設置設備（GPU or CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 生成器類別（需與訓練時一致）
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

# 載入生成器模型
G_A = Generator().to(device)
G_B = Generator().to(device)

G_A.load_state_dict(torch.load("G_A.pth", map_location=device))
G_B.load_state_dict(torch.load("G_B.pth", map_location=device))

G_A.eval()
G_B.eval()

# 圖片預處理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 與訓練時一致
])

# 圖片後處理（將張量轉回圖片）
def tensor_to_image(tensor):
    tensor = tensor.detach().cpu().squeeze(0)
    tensor = tensor * 0.5 + 0.5  # 去標準化
    tensor = torch.clamp(tensor, 0, 1)
    return transforms.ToPILImage()(tensor)

# 載入圖片並轉換為張量
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # 增加 batch 維度
    return image

# 測試函數
def test(real_image_path, fake_image_path):
    real_image = load_image(real_image_path)
    fake_image = load_image(fake_image_path)

    # 將 real_image 轉換為 fake
    generated_fake = G_A(real_image)
    # 將 fake_image 轉換為 real（重建）
    generated_real = G_B(fake_image)

    # 將結果轉為圖片並儲存
    fake_result = tensor_to_image(generated_fake)
    real_result = tensor_to_image(generated_real)

    fake_result.save("generated_fake.jpg")
    real_result.save("reconstructed_real.jpg")

    print("生成的 fake 圖片已儲存為 generated_fake.jpg")
    print("重建的 real 圖片已儲存為 reconstructed_real.jpg")

# 測試模型
test("D:/Jalen/AI_model/style-transfer/my_cyclegan_dataset/real/images/000178.jpg",
     "D:/Jalen/AI_model/style-transfer/my_cyclegan_dataset/fake/images/2_2000.jpg")
