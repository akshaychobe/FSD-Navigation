# src/perception/cone_detector.py

import cv2
import torch
import numpy as np
from src.config import YOLO_MODEL_PATH, CONE_CLASSES, DETECTION_CONFIDENCE_THRESHOLD

class ConeDetector:
    def __init__(self, model_path=YOLO_MODEL_PATH, confidence_threshold=DETECTION_CONFIDENCE_THRESHOLD):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
        self.model.conf = confidence_threshold
        self.classes = CONE_CLASSES

    def detect_cones(self, image):
        results = self.model(image)
        detections = results.xyxy[0].cpu().numpy()

        cones = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if conf >= self.model.conf:
                class_name = self.classes[int(cls)]
                cones.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": float(conf),
                    "class_id": int(cls),
                    "class_name": class_name
                })
        return cones
