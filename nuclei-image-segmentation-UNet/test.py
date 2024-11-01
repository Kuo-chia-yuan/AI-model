import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from main import UNet  # 假設 UNet 模型定義在 main.py 中

# 參數設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "unet_model.pth"  # 已訓練模型的路徑
test_images_dir = "demo_images"  # 測試影像的資料夾
output_dir = "demo_masks"  # 預測結果儲存的資料夾
os.makedirs(output_dir, exist_ok=True)

# 載入模型
model = UNet(in_channels=3, out_channels=1)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# 圖片轉換
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # 調整大小為模型預期的輸入大小
    transforms.ToTensor()
])

# 預測並儲存結果
with torch.no_grad():
    for img_name in os.listdir(test_images_dir):
        # 讀取並轉換影像
        img_path = os.path.join(test_images_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)  # 增加 batch 維度

        # 推論
        output = model(input_tensor)
        predicted_mask = (output > 0.5).float().squeeze(0)  # 二值化並移除 batch 維度

        # 將預測結果轉換為 PIL 圖像並儲存
        mask_image = transforms.ToPILImage()(predicted_mask.cpu())
        output_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_mask.png")
        mask_image.save(output_path)

        print(f"Saved predicted mask for {img_name} to {output_path}")

print("所有影像的預測掩膜已儲存完成！")
