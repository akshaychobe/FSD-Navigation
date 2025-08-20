import os
import cv2
from pathlib import Path
import yaml
from collections import defaultdict

# === CONFIG ===
dataset_dir = r"C:\Users\Lenovo\Github\FSD-Navigation\data\fc-reali-fscoco-2.v2i.yolov5pytorch"
root_dir = r"C:\Users\Lenovo\Github\FSD-Navigation"
splits = ['train', 'valid', 'test']
output_dir = os.path.join(root_dir, 'src' , 'eda', 'eda_outputs' , '05_verify_class_ids')
os.makedirs(output_dir, exist_ok=True)

# === Load class names from data.yaml ===
yaml_path = os.path.join(dataset_dir, 'data.yaml')
with open(yaml_path, 'r') as f:
    data_yaml = yaml.safe_load(f)
names = data_yaml['names']
num_classes = len(names)

# === Collect only fully valid samples per class ===
class_to_samples = defaultdict(list)
for split in splits:
    img_dir = os.path.join(dataset_dir, split, 'images')
    lbl_dir = os.path.join(dataset_dir, split, 'labels')

    for label_file in os.listdir(lbl_dir):
        if not label_file.endswith('.txt'):
            continue

        lbl_path = os.path.join(lbl_dir, label_file)
        img_path = os.path.join(img_dir, label_file.replace('.txt', '.jpg'))
        if not os.path.exists(img_path):
            continue

        with open(lbl_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        class_ids = [int(line.split()[0]) for line in lines if len(line.split()) == 5]

        # Only keep if all labels are the same class
        if len(set(class_ids)) == 1:
            single_class_id = class_ids[0]
            class_to_samples[single_class_id].append((img_path, lbl_path))

# === Visualize ONE clean image per class ===
for class_id in range(num_classes):
    if class_id not in class_to_samples:
        print(f"[WARNING] No valid sample found for class ID {class_id}")
        continue

    img_path, lbl_path = class_to_samples[class_id][0]
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]

    class_name = names[class_id]
    header = f"Class ID: {class_id} | Name: {class_name}"
    cv2.putText(img, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    with open(lbl_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        cid, x, y, bw, bh = map(float, parts)
        cid = int(cid)

        if cid != class_id:
            continue  # Skip other classes

        label = names[cid]
        cx, cy = int(x * w), int(y * h)
        bw, bh = int(bw * w), int(bh * h)
        x1, y1 = cx - bw // 2, cy - bh // 2
        x2, y2 = cx + bw // 2, cy + bh // 2

        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    out_path = os.path.join(output_dir, f"class_check_{class_id}_{class_name}.png")
    cv2.imwrite(out_path, img)

print(f"[INFO] ✅ Verified images saved to: {output_dir}")
