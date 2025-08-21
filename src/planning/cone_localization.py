# cone_localization.py

import cv2
import numpy as np
import os
from ultralytics import YOLO

# === CONFIG ===
MODEL_PATH = "runs/train/exp/weights/best.pt"  # adjust if needed
INPUT_DIR = "outputs/video_frames"
OUTPUT_DIR = "outputs/frames_with_cones"
NPZ_OUTPUT = "outputs/cone_data/cone_coords.npz"
CONF_THRESHOLD = 0.25

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(NPZ_OUTPUT), exist_ok=True)

# === LOAD MODEL ===
model = YOLO(MODEL_PATH)
print(f"[INFO] ✅ Model loaded from {MODEL_PATH}")

cone_data = {}

# === PROCESS FRAMES ===
frame_files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".jpg"))
for fname in frame_files:
    path = os.path.join(INPUT_DIR, fname)
    image = cv2.imread(path)
    detections = model(image, conf=CONF_THRESHOLD)[0]

    frame_cones = []

    for box in detections.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # Only keep blue (0) and yellow (4) cones
        if cls_id in [0, 4]:
            frame_cones.append([cls_id, cx, cy])
            color = (255, 0, 0) if cls_id == 0 else (0, 255, 255)  # blue or yellow
            cv2.circle(image, (cx, cy), 6, color, -1)

    cone_data[fname] = np.array(frame_cones)
    cv2.imwrite(os.path.join(OUTPUT_DIR, fname), image)

    print(f"[{fname}] Saved {len(frame_cones)} cones")

# === SAVE RESULTS ===
np.savez_compressed(NPZ_OUTPUT, **cone_data)
print(f"[INFO] ✅ Cone coordinates saved to {NPZ_OUTPUT}")
