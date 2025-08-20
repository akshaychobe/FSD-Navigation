# 04_image_label_checker.py

import matplotlib.pyplot as plt
from pathlib import Path

base_dir = Path("data/fc-reali-fscoco-2.v2i.yolov5pytorch")
output_path = Path("src/eda/eda_outputs")
output_path.mkdir(parents=True, exist_ok=True)

splits = ['train', 'valid', 'test']
matched_counts = []
missing_labels = []
missing_images = []

for split in splits:
    img_dir = base_dir / split / "images"
    label_dir = base_dir / split / "labels"

    image_files = {f.stem for f in img_dir.glob("*.jpg")} | {f.stem for f in img_dir.glob("*.png")}
    label_files = {f.stem for f in label_dir.glob("*.txt")}

    matched = sorted(image_files & label_files)
    unmatched_images = sorted(image_files - label_files)
    unmatched_labels = sorted(label_files - image_files)

    matched_counts.append(len(matched))
    missing_labels.append(len(unmatched_images))
    missing_images.append(len(unmatched_labels))

# Plotting
x = range(len(splits))
bar_width = 0.25

fig, ax = plt.subplots(figsize=(8, 6))

bars1 = ax.bar([i - bar_width for i in x], matched_counts, width=bar_width, label="Images with Labels", color='mediumseagreen')
bars2 = ax.bar(x, missing_labels, width=bar_width, label="Images without Labels", color='tomato')
bars3 = ax.bar([i + bar_width for i in x], missing_images, width=bar_width, label="Labels without Images", color='dodgerblue')

# Add exact values on top of bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

ax.set_xticks(list(x))
ax.set_xticklabels(splits)
ax.set_ylabel("File Count")
ax.set_title("Matched and Unmatched Image/Label Files by Split")
ax.legend()
plt.tight_layout()
plt.savefig(output_path / "04_image_label_checker.png")
