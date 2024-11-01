import os
import json

def count_images_in_folder(folder_path, extensions=['.jpg', '.jpeg', '.png']):
    """計算資料夾中符合副檔名的圖像文件數量"""
    image_count = 0
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in extensions):
            image_count += 1
    return image_count

def count_images_and_annotations_in_json(json_path):
    """計算 .json 文件中圖像和標註的數量"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 計算圖像和標註數量
    image_count = len(data['images'])
    annotation_count = len(data['annotations'])
    
    return image_count, annotation_count

# 設定資料夾和標註文件的路徑
train_folder = "D:/Jalen/AI_model/YOLO/train2017"
val_folder = "D:/Jalen/AI_model/YOLO/val2017"
train_json = "D:/Jalen/AI_model/YOLO/annotations/instances_train2017_filtered.json"
val_json = "D:/Jalen/AI_model/YOLO/annotations/instances_val2017_filtered.json"

# 計算圖像數量
train_image_count = count_images_in_folder(train_folder)
val_image_count = count_images_in_folder(val_folder)

# 計算標註文件中的圖像和標註數量
train_json_image_count, train_annotation_count = count_images_and_annotations_in_json(train_json)
val_json_image_count, val_annotation_count = count_images_and_annotations_in_json(val_json)

# 印出結果
print(f"Number of images in train2017 folder: {train_image_count}")
print(f"Number of images in val2017 folder: {val_image_count}")
print(f"Number of images in instances_train2017_filtered.json: {train_json_image_count}")
print(f"Number of annotations in instances_train2017_filtered.json: {train_annotation_count}")
print(f"Number of images in instances_val2017_filtered.json: {val_json_image_count}")
print(f"Number of annotations in instances_val2017_filtered.json: {val_annotation_count}")
