import os
from ultralytics import YOLO

def main():
    # Load model
    model = YOLO("yolov5n.pt")  # Use pretrained YOLOv5 Nano model

    # Start training with specified parameters
    model.train(
        data=r"data/data.yaml",
        epochs=100,
        imgsz=512,
        batch=4,
        workers=2,
        lr0=0.01,
        optimizer="SGD",
        cos_lr=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        mosaic=0.0,
        shear=0.0,
        mixup=0.0,
        copy_paste=0.0,
        device=0,  # GPU
        project="runs/train",
        name="fsd_custom_yolov5",
        exist_ok=True
    )

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
