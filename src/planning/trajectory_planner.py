import os
import numpy as np
import cv2

# === CONFIG ===
INPUT_DIR = "outputs/frames_with_cones"
LOCALIZED_FILE = "outputs/cone_data/cone_coords.npz"
OUTPUT_DIR = "outputs/trajectory_visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD COORDINATE DATA ===
data = np.load(LOCALIZED_FILE, allow_pickle=True)

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

for filename in sorted(data.files):
    detections = data[filename]
    frame_path = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(frame_path):
        print(f"[WARNING] Frame not found: {frame_path}")
        continue

    frame = cv2.imread(frame_path)
    h, w = frame.shape[:2]
    ROI_POLYGON = new_func(h, w)

    # Car fixed top point (approx center top of car hood)
    car_pt = (int(w / 2), int(0.75 * h))

    # === Filter cones in ROI ===
    left_cones = [(x, y) for cls, x, y in detections if cls == 0 and inside_roi(x, y, ROI_POLYGON)]
    right_cones = [(x, y) for cls, x, y in detections if cls == 4 and inside_roi(x, y, ROI_POLYGON)]

    midpoints = pair_cones(left_cones, right_cones)
    midpoints_in_roi = [pt for pt in midpoints if inside_roi(pt[0], pt[1], ROI_POLYGON)]

    if midpoints_in_roi:
        last_midpoints = midpoints_in_roi
    else:
        midpoints_in_roi = last_midpoints

    # === Draw ROI ===
    cv2.polylines(frame, [ROI_POLYGON], isClosed=True, color=(0, 255, 0), thickness=2)

    # === Draw cones ===
    for x, y in left_cones:
        cv2.circle(frame, (int(x), int(y)), 6, (255, 0, 0), -1)
    for x, y in right_cones:
        cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 255), -1)

    # === Draw midpoints and arrows ===
    for x, y in midpoints_in_roi:
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    if midpoints_in_roi:
        # Draw arrow from car to nearest midpoint
        cv2.arrowedLine(frame, car_pt, midpoints_in_roi[0], (0, 255, 0), 2, tipLength=0.2)
        if len(midpoints_in_roi) > 1:
            cv2.line(frame, midpoints_in_roi[0], midpoints_in_roi[1], (0, 255, 0), 2)

    # === Draw car fixed point ===
    cv2.circle(frame, car_pt, 5, (0, 255, 0), -1)

    # === Text ===
    text = f"{filename}: {len(left_cones)} blue, {len(right_cones)} yellow, {len(midpoints_in_roi)} pairs"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 255, 180), 2)

    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), frame)

print("[INFO] ✅ Trajectory images saved in:", OUTPUT_DIR)
