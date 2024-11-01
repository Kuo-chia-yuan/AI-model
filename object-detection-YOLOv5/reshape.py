import os

labels_dir = 'datasets/labels/val'  # 標註文件資料夾
max_class_id = 79  # 設定最大類別 ID（根據 nc-1）

for filename in os.listdir(labels_dir):
    if not filename.endswith('.txt'):
        continue

    file_path = os.path.join(labels_dir, filename)
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 過濾出範圍內的標註行
    new_lines = []
    for line in lines:
        class_id = int(line.split()[0])  # 取得類別 ID
        if 0 <= class_id <= max_class_id:
            new_lines.append(line)
        else:
            print(f"Warning: {filename} contains invalid class ID {class_id}")

    # 覆寫文件，僅保留範圍內的標註行
    with open(file_path, 'w') as f:
        f.writelines(new_lines)
