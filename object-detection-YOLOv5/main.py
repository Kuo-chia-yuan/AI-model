from ultralytics import YOLO

def main():
    # 加載 YOLO 模型（可以使用 yolov5s.pt 或 yolov8s.pt 等模型）
    model = YOLO('yolov5s.pt')  # 或 'path/to/your/model.pt'

    # 開始訓練
    model.train(
        data='custom_coco.yaml',  # 資料集配置文件
        epochs=10,                # 訓練次數
        imgsz=640,                # 圖片尺寸
        batch=16,                 # 批次大小
        name='yolo_model'         # 訓練結果的保存名稱
    )

if __name__ == "__main__":
    main()