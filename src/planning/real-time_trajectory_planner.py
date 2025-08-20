import os
import numpy as np
import cv2
from ultralytics import YOLO

# === CONFIG ===
INPUT_DIR = "outputs/frames_with_cones"
OUTPUT_DIR = "outputs/trajectory_visuals_realtime"
MODEL_PATH = r"runs\train\exp\weights\best.pt"  # Replace with actual YOLO model path
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD YOLO MODEL ===
model = YOLO(MODEL_PATH)

# === Midpoint History ===
last_midpoints = []

def inside_roi(x, y, polygon):
    x = float(x)
    y = float(y)
    return cv2.pointPolygonTest(polygon, (x, y), True) >= -3

def pair_cones(left_cones, right_cones, y_threshold=30):
    midpoints = []
    for lx, ly in left_cones:
        best_match = None
        min_y_diff = y_threshold
        for rx, ry in right_cones:
            if abs(ly - ry) < min_y_diff:
                best_match = (rx, ry)
                min_y_diff = abs(ly - ry)
        if best_match:
            mx = int((lx + best_match[0]) / 2)
            my = int((ly + best_match[1]) / 2)
            midpoints.append((mx, my))
    return midpoints

def new_func(h, w):
    ROI_POLYGON = np.array([
        [int(0.05 * w), int(0.98 * h)],
        [int(0.35 * w), int(0.45 * h)],
        [int(0.65 * w), int(0.45 * h)],
        [int(0.95 * w), int(0.98 * h)]
    ], np.int32)
    return ROI_POLYGON

# === MAIN LOOP ===
for filename in sorted(os.listdir(INPUT_DIR)):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    frame_path = os.path.join(INPUT_DIR, filename)
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"[WARNING] Could not read frame: {frame_path}")
        continue

    h, w = frame.shape[:2]
    ROI_POLYGON = new_func(h, w)

    # Fixed car-top reference point
    car_pt = (int(w / 2), int(0.75 * h))

    # Run YOLO on this frame
    results = model(frame, verbose=False)[0]
    detections = []
    for box in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        detections.append((int(cls), cx, cy))

    # Filter cones
    left_cones = [(x, y) for cls, x, y in detections if cls == 0 and inside_roi(x, y, ROI_POLYGON)]
    right_cones = [(x, y) for cls, x, y in detections if cls == 4 and inside_roi(x, y, ROI_POLYGON)]

    midpoints = pair_cones(left_cones, right_cones)
    midpoints_in_roi = [pt for pt in midpoints if inside_roi(pt[0], pt[1], ROI_POLYGON)]

    if midpoints_in_roi:
        last_midpoints = midpoints_in_roi
    else:
        midpoints_in_roi = last_midpoints

    # Draw ROI
    cv2.polylines(frame, [ROI_POLYGON], isClosed=True, color=(0, 255, 0), thickness=2)

    # Draw cones
    for x, y in left_cones:
        cv2.circle(frame, (x, y), 6, (255, 0, 0), -1)
    for x, y in right_cones:
        cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)

    # Draw midpoints
    for x, y in midpoints_in_roi:
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    # Draw trajectory from car_pt to nearest midpoint and optional second
    if midpoints_in_roi:
        dx = midpoints_in_roi[0][0] - car_pt[0]
        dy = midpoints_in_roi[0][1] - car_pt[1]
        shrink_factor = 0.15
        px = int(midpoints_in_roi[0][0] - dx * shrink_factor)
        py = int(midpoints_in_roi[0][1] - dy * shrink_factor)
        cv2.arrowedLine(frame, car_pt, (px, py), (0, 255, 0), thickness=4, tipLength=0.25)

        if len(midpoints_in_roi) > 1:
            cv2.line(frame, midpoints_in_roi[0], midpoints_in_roi[1], (0, 255, 0), 3)

    # Draw quadratic curve if ≥3 midpoints in ROI
    if len(midpoints_in_roi) >= 3:
        midpoints_in_roi = sorted(midpoints_in_roi, key=lambda pt: pt[1])
        x_vals = np.array([pt[0] for pt in midpoints_in_roi])
        y_vals = np.array([pt[1] for pt in midpoints_in_roi])

        coeffs = np.polyfit(y_vals, x_vals, deg=2)
        y_fit = np.linspace(y_vals[0], y_vals[-1], 100)
        x_fit = np.polyval(coeffs, y_fit)
        curve_pts = np.array([[int(x), int(y)] for x, y in zip(x_fit, y_fit)], dtype=np.int32)
        cv2.polylines(frame, [curve_pts], isClosed=False, color=(0, 255, 255), thickness=2)

    # Draw car point
    cv2.circle(frame, car_pt, 6, (0, 255, 0), -1)

    # Text overlay
    text = f"{filename}: {len(left_cones)} blue, {len(right_cones)} yellow, {len(midpoints_in_roi)} pairs"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 255, 180), 2)

    # Save frame
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), frame)

print("[INFO] ✅ Trajectory images saved in:", OUTPUT_DIR)
