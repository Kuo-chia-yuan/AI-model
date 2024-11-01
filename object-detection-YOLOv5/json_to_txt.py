import json
import os

# 指定 .json 文件和輸出目錄
json_file = 'instances_val.json'  # 你的 COCO 格式的 .json 文件路徑
output_dir = 'labels/val'  # 轉換後 .txt 文件儲存的路徑
os.makedirs(output_dir, exist_ok=True)

# 讀取 JSON 文件
with open(json_file, 'r') as f:
    data = json.load(f)

# 建立字典以便快速查找圖片大小
image_id_to_info = {image['id']: (image['file_name'], image['width'], image['height']) for image in data['images']}

# 轉換每個標註
for annotation in data['annotations']:
    image_id = annotation['image_id']
    category_id = annotation['category_id']
    bbox = annotation['bbox']  # [x_min, y_min, width, height]

    # 取得圖片資訊
    file_name, img_width, img_height = image_id_to_info[image_id]

    # 計算 YOLO 格式的中心點和寬高（相對比例）
    x_min, y_min, bbox_width, bbox_height = bbox
    x_center = (x_min + bbox_width / 2) / img_width
    y_center = (y_min + bbox_height / 2) / img_height
    width = bbox_width / img_width
    height = bbox_height / img_height

    # 準備 YOLO 格式的標註
    class_id = category_id - 1  # 根據需要調整類別 ID
    yolo_annotation = f"{class_id} {x_center} {y_center} {width} {height}\n"

    # 輸出成 .txt 文件，文件名與圖片名相同
    txt_filename = os.path.join(output_dir, os.path.splitext(file_name)[0] + '.txt')
    with open(txt_filename, 'a') as txt_file:  # 使用 'a'，以追加方式寫入
        txt_file.write(yolo_annotation)

print("標註已成功轉換成 YOLO 格式！")
