from ultralytics import YOLO
import cv2
import os

# === Config ===
model = YOLO("runs/train/exp/weights/best.pt")
input_dir = r"C:\Users\Lenovo\Github\FSD-Navigation\data\test\images"
output_img_dir = r"results/test_inference/images"
output_lbl_dir = r"results/test_inference/labels"

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_lbl_dir, exist_ok=True)

# === Inference and Save ===
for file in os.listdir(input_dir):
    if not file.endswith(".jpg"):
        continue

    img_path = os.path.join(input_dir, file)
    results = model(img_path)[0]  # Only get first result

    # Save annotated image
    annotated_img = results.plot()
    cv2.imwrite(os.path.join(output_img_dir, file), annotated_img)

    # Save YOLO-format predictions
    label_lines = []
    for box in results.boxes.data.cpu().numpy():
        cls_id = int(box[5])
        conf = box[4]
        x_center, y_center, width, height = box[0:4]

        # Normalize coordinates (0-1)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        x_center /= w
        y_center /= h
        width /= w
        height /= h

        line = f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.4f}"
        label_lines.append(line)

    label_filename = file.replace(".jpg", ".txt")
    with open(os.path.join(output_lbl_dir, label_filename), "w") as f:
        f.write("\n".join(label_lines))

print("[INFO] ✅ Inference complete. Images and labels saved.")
