# 03_label_visualization.py

import matplotlib.pyplot as plt
from pathlib import Path

base_dir = Path("data/fc-reali-fscoco-2.v2i.yolov5pytorch")
output_path = Path("src/eda/eda_outputs")
output_path.mkdir(parents=True, exist_ok=True)

widths, heights = [], []

for split in ['train', 'valid', 'test']:
    label_dir = base_dir / split / "labels"
    for txt_file in label_dir.glob("*.txt"):
        with open(txt_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, _, _, w, h = map(float, parts)
                    widths.append(w)
                    heights.append(h)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(widths, heights, alpha=0.3, s=10, color="purple")
plt.title("BBox Width vs Height (Normalized)")
plt.xlabel("Width")
plt.ylabel("Height")
plt.grid(True)
plt.tight_layout()
plt.savefig(output_path / "03_label_dimensions_scatter.png")
