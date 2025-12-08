"""
CLIP 模型推理 - 圖片分類（修正版）
解決 PyTorch 2.6 的 weights_only 問題
"""

# ==================== 導入套件 ====================

import torch
import torch.nn.functional as F
from PIL import Image
import clip
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import os
import warnings

# 添加安全的全局變數（解決 numpy 問題）
try:
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
except:
    pass

warnings.filterwarnings('ignore')

print("✅ 套件導入完成")

# ==================== 載入模型（修正版）====================

def load_trained_model(checkpoint_path='best_clip_rn50_model.pt', device=None):
    """
    載入訓練好的 CLIP 模型（修正 PyTorch 2.6 兼容性）

    Args:
        checkpoint_path: 模型檔案路徑
        device: 運算設備 (None 則自動選擇)

    Returns:
        model: 載入的模型
        preprocess: 圖片預處理函數
        device: 使用的設備
    """
    print("=" * 70)
    print("載入訓練好的模型")
    print("=" * 70)

    # 設備
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"✅ 使用設備: {device}")

    # 載入 CLIP 模型架構
    print("✅ 載入 CLIP RN50 架構...")
    model, preprocess = clip.load("RN50", device=device, jit=False)

    # 載入訓練好的權重
    if os.path.exists(checkpoint_path):
        print(f"✅ 找到模型檔案: {checkpoint_path}")

        try:
            # 方法 1: 嘗試使用 weights_only=True（安全）
            print("   嘗試安全載入模式...")
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=True
            )
            print("   ✅ 使用安全模式載入成功")

        except Exception as e1:
            print(f"   ⚠️  安全模式失敗: {str(e1)[:100]}")

            try:
                # 方法 2: 使用 weights_only=False（如果你信任這個檔案）
                print("   嘗試兼容模式...")
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location=device,
                    weights_only=False  # ← 關鍵修改
                )
                print("   ✅ 使用兼容模式載入成功")

            except Exception as e2:
                print(f"   ❌ 兼容模式也失敗: {str(e2)[:100]}")
                print(f"   ⚠️  使用預訓練的 CLIP 模型")
                model.eval()
                return model, preprocess, device

        # 載入權重
        try:
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ 載入模型權重")

                # 顯示訓練資訊
                if 'epoch' in checkpoint:
                    print(f"   Epoch: {checkpoint['epoch']}")
                if 'val_loss' in checkpoint:
                    print(f"   Val Loss: {checkpoint['val_loss']:.4f}")
                if 'val_acc' in checkpoint:
                    print(f"   Val Acc: {checkpoint['val_acc']:.2f}%")
            else:
                model.load_state_dict(checkpoint)
                print(f"✅ 載入模型權重")

        except Exception as e:
            print(f"   ❌ 載入權重失敗: {e}")
            print(f"   ⚠️  使用預訓練的 CLIP 模型")
    else:
        print(f"⚠️  找不到模型檔案: {checkpoint_path}")
        print(f"⚠️  使用預訓練的 CLIP 模型")

    # 設為評估模式
    model.eval()

    print(f"✅ 模型載入完成\n")

    return model, preprocess, device

# ==================== 圖片分類函數 ====================

def classify_image(
    image_path: str,
    categories: List[str],
    model,
    preprocess,
    device,
    top_k: int = None,
    show_image: bool = True
) -> Dict[str, float]:
    """
    對圖片進行分類

    Args:
        image_path: 圖片路徑
        categories: 分類類別列表
        model: CLIP 模型
        preprocess: 預處理函數
        device: 運算設備
        top_k: 顯示前 k 個最高機率的類別 (None 則顯示全部)
        show_image: 是否顯示圖片

    Returns:
        results: {類別: 機率} 的字典
    """
    print("=" * 70)
    print("圖片分類")
    print("=" * 70)

    # 載入圖片
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到圖片: {image_path}")

    print(f"✅ 載入圖片: {image_path}")

    image = Image.open(image_path).convert('RGB')
    original_image = image.copy()

    # 預處理圖片
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # 準備文字描述
    # 使用 "a photo of {category}" 格式可以提升準確度
    text_descriptions = [f"a photo of {category}" for category in categories]
    text_tokens = clip.tokenize(text_descriptions).to(device)

    print(f"✅ 分類類別數量: {len(categories)}")
    print(f"✅ 類別: {', '.join(categories)}\n")

    # 推理
    with torch.no_grad():
        # 編碼圖片和文字
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)

        # 正規化
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        # 計算相似度
        logit_scale = model.logit_scale.exp()
        logits = logit_scale * (image_features @ text_features.T)

        # 計算機率 (softmax)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

    # 整理結果
    results = {category: float(prob) for category, prob in zip(categories, probs)}

    # 排序
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    # 顯示結果
    print("分類結果:")
    print("-" * 70)

    if top_k is not None:
        display_results = dict(list(sorted_results.items())[:top_k])
    else:
        display_results = sorted_results

    for i, (category, prob) in enumerate(display_results.items(), 1):
        bar = '█' * int(prob * 50)
        print(f"{i:2d}. {category:30s} {prob*100:6.2f}% {bar}")

    print("-" * 70)

    # 顯示圖片和結果
    if show_image:
        visualize_results(original_image, sorted_results, top_k)

    return sorted_results

# ==================== 視覺化結果 ====================

def visualize_results(image, results: Dict[str, float], top_k: int = 5):
    """
    視覺化分類結果

    Args:
        image: PIL Image
        results: 分類結果字典
        top_k: 顯示前 k 個結果
    """
    # 取前 k 個結果
    if top_k is not None:
        results = dict(list(results.items())[:top_k])

    # 創建圖表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 顯示圖片
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('Input Image', fontsize=14, fontweight='bold')

    # 顯示機率條形圖
    categories = list(results.keys())
    probabilities = list(results.values())

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(categories)))
    bars = ax2.barh(categories, probabilities, color=colors)

    # 在條形上顯示數值
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2,
                f'{prob*100:.1f}%',
                ha='left', va='center', fontsize=10, fontweight='bold')

    ax2.set_xlabel('Probability', fontsize=12)
    ax2.set_title('Classification Results', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1.0)
    ax2.grid(axis='x', alpha=0.3)

    # 反轉 y 軸（最高機率在上）
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig('classification_result.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ 結果圖片已儲存: classification_result.png")
    plt.show()

# ==================== 批次分類 ====================

def classify_multiple_images(
    image_paths: List[str],
    categories: List[str],
    model,
    preprocess,
    device,
    top_k: int = 3
):
    """
    對多張圖片進行分類

    Args:
        image_paths: 圖片路徑列表
        categories: 分類類別列表
        model: CLIP 模型
        preprocess: 預處理函數
        device: 運算設備
        top_k: 顯示前 k 個結果
    """
    print("=" * 70)
    print(f"批次分類 ({len(image_paths)} 張圖片)")
    print("=" * 70)

    all_results = []

    for i, image_path in enumerate(image_paths, 1):
        print(f"\n處理圖片 {i}/{len(image_paths)}: {image_path}")
        print("-" * 70)

        try:
            results = classify_image(
                image_path,
                categories,
                model,
                preprocess,
                device,
                top_k=top_k,
                show_image=False
            )
            all_results.append({
                'image_path': image_path,
                'results': results
            })
        except Exception as e:
            print(f"❌ 處理失敗: {e}")

    # 視覺化所有結果
    if all_results:
        visualize_batch_results(all_results, top_k)

    return all_results

def visualize_batch_results(all_results, top_k: int = 3):
    """視覺化批次分類結果"""
    n_images = len(all_results)

    fig, axes = plt.subplots(n_images, 2, figsize=(14, 5 * n_images))

    if n_images == 1:
        axes = [axes]

    for i, result in enumerate(all_results):
        image_path = result['image_path']
        results = result['results']

        # 載入圖片
        image = Image.open(image_path).convert('RGB')

        # 顯示圖片
        axes[i][0].imshow(image)
        axes[i][0].axis('off')
        axes[i][0].set_title(f'Image {i+1}: {os.path.basename(image_path)}',
                            fontsize=12, fontweight='bold')

        # 顯示結果
        top_results = dict(list(results.items())[:top_k])
        categories = list(top_results.keys())
        probabilities = list(top_results.values())

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(categories)))
        bars = axes[i][1].barh(categories, probabilities, color=colors)

        for bar, prob in zip(bars, probabilities):
            width = bar.get_width()
            axes[i][1].text(width, bar.get_y() + bar.get_height()/2,
                           f'{prob*100:.1f}%',
                           ha='left', va='center', fontsize=10, fontweight='bold')

        axes[i][1].set_xlabel('Probability', fontsize=11)
        axes[i][1].set_title(f'Top {top_k} Predictions', fontsize=12, fontweight='bold')
        axes[i][1].set_xlim(0, 1.0)
        axes[i][1].grid(axis='x', alpha=0.3)
        axes[i][1].invert_yaxis()

    plt.tight_layout()
    plt.savefig('batch_classification_results.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ 批次結果圖片已儲存: batch_classification_results.png")
    plt.show()

# ==================== 互動式分類 ====================

def interactive_classification(model, preprocess, device):
    """
    互動式分類介面
    """
    print("=" * 70)
    print("互動式圖片分類")
    print("=" * 70)
    print("\n使用說明:")
    print("  1. 輸入圖片路徑")
    print("  2. 輸入分類類別（用逗號分隔）")
    print("  3. 查看分類結果")
    print("  4. 輸入 'quit' 退出\n")

    while True:
        print("-" * 70)

        # 輸入圖片路徑
        image_path = input("請輸入圖片路徑 (或輸入 'quit' 退出): ").strip()

        if image_path.lower() == 'quit':
            print("👋 再見！")
            break

        if not os.path.exists(image_path):
            print(f"❌ 找不到圖片: {image_path}")
            continue

        # 輸入分類類別
        categories_input = input("請輸入分類類別（用逗號分隔）: ").strip()
        categories = [cat.strip() for cat in categories_input.split(',') if cat.strip()]

        if not categories:
            print("❌ 請至少輸入一個類別")
            continue

        # 執行分類
        try:
            results = classify_image(
                image_path,
                categories,
                model,
                preprocess,
                device,
                top_k=5,
                show_image=True
            )

            # 詢問是否繼續
            continue_input = input("\n繼續分類其他圖片？(y/n): ").strip().lower()
            if continue_input != 'y':
                print("👋 再見！")
                break

        except Exception as e:
            print(f"❌ 分類失敗: {e}")

# ==================== 主程式 ====================

def main():
    """主程式"""

    print("\n" + "=" * 70)
    print("CLIP 圖片分類系統")
    print("=" * 70)

    # 載入模型
    model, preprocess, device = load_trained_model('best_clip_rn50_model.pt')

    # 選擇模式
    print("\n請選擇模式:")
    print("  1. 單張圖片分類")
    print("  2. 批次圖片分類")
    print("  3. 互動式分類")

    mode = input("\n請輸入模式 (1/2/3): ").strip()

    if mode == '1':
        # 單張圖片分類
        print("\n" + "=" * 70)
        print("單張圖片分類")
        print("=" * 70)

        # 範例
        image_path = input("請輸入圖片路徑: ").strip()

        categories_input = input("請輸入分類類別（用逗號分隔）: ").strip()
        categories = [cat.strip() for cat in categories_input.split(',') if cat.strip()]

        if not categories:
            print("❌ 請至少輸入一個類別")
            return

        results = classify_image(
            image_path,
            categories,
            model,
            preprocess,
            device,
            top_k=5,
            show_image=True
        )

    elif mode == '2':
        # 批次分類
        print("\n" + "=" * 70)
        print("批次圖片分類")
        print("=" * 70)

        # 輸入圖片路徑
        print("\n請輸入圖片路徑（每行一個，輸入空行結束）:")
        image_paths = []
        while True:
            path = input().strip()
            if not path:
                break
            image_paths.append(path)

        if not image_paths:
            print("❌ 請至少輸入一個圖片路徑")
            return

        categories_input = input("\n請輸入分類類別（用逗號分隔）: ").strip()
        categories = [cat.strip() for cat in categories_input.split(',') if cat.strip()]

        if not categories:
            print("❌ 請至少輸入一個類別")
            return

        results = classify_multiple_images(
            image_paths,
            categories,
            model,
            preprocess,
            device,
            top_k=3
        )

    elif mode == '3':
        # 互動式分類
        interactive_classification(model, preprocess, device)

    else:
        print("❌ 無效的模式選擇")

    print("\n" + "=" * 70)
    print("✅ 程式執行完成")
    print("=" * 70)

# ==================== 快速使用範例 ====================

def quick_example():
    """
    快速使用範例
    """
    print("=" * 70)
    print("快速使用範例")
    print("=" * 70)

    # 載入模型
    model, preprocess, device = load_trained_model('best_clip_rn50_model.pt')

    # 範例: 動物分類
    print("\n範例: 動物分類")
    print("-" * 70)

    # 請替換成你的圖片路徑
    image_path = input("請輸入測試圖片路徑: ").strip()

    if os.path.exists(image_path):
        results = classify_image(
            image_path=image_path,
            categories=['cat', 'dog', 'bird', 'fish', 'horse', 'cow', 'sheep', 'rabbit'],
            model=model,
            preprocess=preprocess,
            device=device,
            top_k=5,
            show_image=True
        )

        # 顯示最佳預測
        best_category = max(results, key=results.get)
        best_prob = results[best_category]
        print(f"\n🎯 最佳預測: {best_category} ({best_prob*100:.2f}%)")
    else:
        print(f"❌ 找不到圖片: {image_path}")

# ==================== 執行 ====================

if __name__ == "__main__":
    # 方式 1: 執行主程式（互動式）
    main()

    # 方式 2: 快速範例（適合測試）
    # quick_example()

    # 方式 3: 直接使用（適合 Jupyter Notebook）
    # model, preprocess, device = load_trained_model('best_clip_rn50_model.pt')
    # results = classify_image(
    #     'your_image.jpg',
    #     ['category1', 'category2', 'category3'],
    #     model, preprocess, device
    # )
