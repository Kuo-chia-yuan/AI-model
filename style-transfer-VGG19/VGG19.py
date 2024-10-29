import torch
from torchvision import models, transforms
from PIL import Image

# 使用 GPU（若可用），否則使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 載入 VGG19 模型，並只取其特徵層部分
vgg = models.vgg19(pretrained=True).features.to(device).eval()

# 鎖定 VGG19 權重，不進行訓練
for param in vgg.parameters():
    param.requires_grad = False

# 圖片轉換：調整大小、標準化、轉為 Tensor
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # VGG19 預訓練模型的標準化
])

# 載入圖片並轉為 Tensor
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # 增加 batch 維度
    return image

content_img = load_image("C:/Users/Jalen/.cache/kagglehub/datasets/awsaf49/coco-2017-dataset/versions/2/coco2017/train2017/000000000009.jpg")
style_img = load_image("C:/Users/Jalen/.cache/kagglehub/datasets/soumikrakshit/images-for-style-transfer/versions/1/Data/Artworks/81842.jpg")

print(f"內容圖片大小: {content_img.shape}")
print(f"風格圖片大小: {style_img.shape}")

# 定義幫助函數：提取指定層的特徵
def get_features(image, model, layers):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[name] = x
    return features

# 層名稱的對應
content_layer = ['21']  # conv4_2
style_layers = ['0', '5', '10', '19', '28']  # conv1_1 到 conv5_1

# 提取特徵
content_features = get_features(content_img, vgg, content_layer)
style_features = get_features(style_img, vgg, style_layers)

# 計算 Gram 矩陣，用於風格損失
def gram_matrix(tensor):
    _, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    return torch.mm(tensor, tensor.t())

# 定義內容損失和風格損失
def content_loss(gen_features, target_features):
    return torch.mean((gen_features - target_features) ** 2)

def style_loss(gen_features, style_gram):
    G = gram_matrix(gen_features)
    return torch.mean((G - style_gram) ** 2)

# 初始化生成圖片為內容圖片的副本
gen_img = content_img.clone().requires_grad_(True)

# 使用 L-BFGS 優化器
optimizer = torch.optim.LBFGS([gen_img])

# 提取風格圖片的 Gram 矩陣
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_layers}

# 訓練過程
steps = 300  # 訓練步數

def closure():
    optimizer.zero_grad()

    # 提取生成圖片的特徵
    gen_features = get_features(gen_img, vgg, content_layer + style_layers)

    # 計算內容損失
    c_loss = content_loss(gen_features['21'], content_features['21'])

    # 計算風格損失
    s_loss = sum(style_loss(gen_features[layer], style_grams[layer]) for layer in style_layers)

    # 總損失
    total_loss = c_loss + 1e6 * s_loss  # 風格損失加權較大
    total_loss.backward()

    return total_loss

# 開始優化
for step in range(steps):
    optimizer.step(closure)
    print(f"Step {step}, Loss: {closure().item()}")

from torchvision.utils import save_image
import matplotlib.pyplot as plt

# 將生成圖片反標準化到 [0, 1] 區間
def unnormalize(tensor):
    tensor = tensor.detach().cpu().squeeze(0)  # 移除 batch 維度
    tensor = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
             torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)  # 反標準化
    return torch.clamp(tensor, 0, 1)  # 限制範圍到 [0, 1]

# 取得最終的生成圖片
final_img = unnormalize(gen_img)

# 顯示生成圖片
plt.imshow(final_img.permute(1, 2, 0))  # 調整維度為 (H, W, C)
plt.axis('off')  # 隱藏座標軸
plt.show()

# 儲存生成圖片為 JPEG 檔案
save_image(final_img, "generated_image.jpg")
print("生成圖片已儲存為 generated_image.jpg")
