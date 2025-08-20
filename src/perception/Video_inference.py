import cv2
import imageio
from ultralytics import YOLO
from pathlib import Path

# === CONFIG ===
video_path = r"C:\Users\Lenovo\Github\FSD-Navigation\Test_videos\Skidpad_FSE.mp4"
model_path = r"C:\Users\Lenovo\Github\FSD-Navigation\runs\train\exp\weights\best.onnx"
gif_output_path = r"C:\Users\Lenovo\Github\FSD-Navigation\Test_videos\onnx_demo_output.gif"

# === Load model ===
model = YOLO(model_path)

# === Read video ===
cap = cv2.VideoCapture(video_path)
frames = []
frame_count = 0
max_frames = 100  # Limit GIF length to avoid large files

while True:
    ret, frame = cap.read()
    if not ret or frame_count >= max_frames:
        break

    # Run YOLO inference
    results = model(frame)
    annotated = results[0].plot()  # Draw bounding boxes

    # Convert BGR to RGB and resize (optional)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    frames.append(annotated_rgb)

    frame_count += 1

cap.release()

# === Save as GIF using imageio ===
imageio.mimsave(gif_output_path, frames, duration=0.05)  # duration in seconds per frame

print(f"[INFO] ✅ GIF saved to: {gif_output_path}")
