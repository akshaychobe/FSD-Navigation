import numpy as np
import cv2
import os

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
    """Return True if point (x, y) lies inside the polygon ROI with buffer margin."""
    x = float(x)
    y = float(y)
    return cv2.pointPolygonTest(polygon, (x, y), True) >= -3


def pair_cones(left_cones, right_cones, y_threshold=30):
    """
    Pair left (blue) and right (yellow) cones by Y alignment, and return midpoints.
    """
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
        [int(0.05 * w), int(0.98 * h)],  # bottom-left corner (near bottom edge)
        [int(0.35 * w), int(0.45 * h)],  # upper-left inward
        [int(0.65 * w), int(0.45 * h)],  # upper-right inward
        [int(0.95 * w), int(0.98 * h)]   # bottom-right corner
    ], np.int32)
    
    return ROI_POLYGON

for filename in sorted(data.files):
    detections = data[filename]  # [(cls_id, cx, cy), ...]

    frame_path = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(frame_path):
        print(f"[WARNING] Frame not found: {frame_path}")
        continue

    frame = cv2.imread(frame_path)
    h, w = frame.shape[:2]

    # === Define Dynamic Trapezoid ROI ===
    ROI_POLYGON = new_func(h, w)


    # === Filter Cones in ROI ===
    left_cones = [(x, y) for cls, x, y in detections if cls == 0 and inside_roi(x, y, ROI_POLYGON)]
    right_cones = [(x, y) for cls, x, y in detections if cls == 4 and inside_roi(x, y, ROI_POLYGON)]

    midpoints = pair_cones(left_cones, right_cones)

    if midpoints:
        last_midpoints = midpoints
    else:
        midpoints = last_midpoints  # fallback if no new pairs

    # === Draw ROI ===
    cv2.polylines(frame, [ROI_POLYGON], isClosed=True, color=(0, 255, 0), thickness=2)

    # === Draw Cones ===
    for x, y in left_cones:
        cv2.circle(frame, (int(x), int(y)), 6, (255, 0, 0), -1)  # Blue
    for x, y in right_cones:
        cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 255), -1)  # Yellow

    # === Draw Midpoints ===
    for x, y in midpoints:
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red midpoint

    # === Debug Text ===
    text = f"{filename}: {len(left_cones)} blue, {len(right_cones)} yellow, {len(midpoints)} pairs"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 255, 180), 2)

    save_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(save_path, frame)

print(f"[INFO] ✅ Trajectory visuals saved in: {OUTPUT_DIR}")

# === Save all midpoints per frame to .npz ===
midpoints_dict = {}

for filename in sorted(data.files):
    detections = data[filename]
    frame_path = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(frame_path):
        continue

    frame = cv2.imread(frame_path)
    h, w = frame.shape[:2]
    ROI_POLYGON = new_func(h, w)

    left_cones = [(x, y) for cls, x, y in detections if cls == 0 and inside_roi(x, y, ROI_POLYGON)]
    right_cones = [(x, y) for cls, x, y in detections if cls == 4 and inside_roi(x, y, ROI_POLYGON)]

    midpoints = pair_cones(left_cones, right_cones)
    if not midpoints:
        midpoints = last_midpoints
    else:
        last_midpoints = midpoints

    midpoints_dict[filename] = midpoints

np.savez_compressed("outputs/midpoints.npz", **midpoints_dict)
print("[INFO] ✅ Midpoints saved to: outputs/midpoints.npz")

