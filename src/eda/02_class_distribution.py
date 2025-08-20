from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

base_dir = Path("data/fc-reali-fscoco-2.v2i.yolov5pytorch")
output_path = Path("src/eda/eda_outputs")
output_path.mkdir(parents=True, exist_ok=True)

with open(base_dir / "data.yaml", "r") as f:
    class_names = yaml.safe_load(f)["names"]

class_counts = defaultdict(int)

for split in ['train', 'valid', 'test']:
    label_dir = base_dir / split / "labels"
    for txt_file in label_dir.glob("*.txt"):
        with open(txt_file, "r") as f:
            for line in f:
                cls_id = int(line.strip().split()[0])
                class_counts[class_names[cls_id]] += 1

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(class_counts.keys(), class_counts.values(), color="steelblue")
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height}', (bar.get_x() + bar.get_width() / 2, height),
                ha='center', va='bottom', fontsize=9)

ax.set_title("Class Distribution Across All Splits")
ax.set_ylabel("Instance Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_path / "02_class_distribution.png")
