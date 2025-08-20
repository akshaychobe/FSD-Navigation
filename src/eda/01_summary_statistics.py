import os
from pathlib import Path
import matplotlib.pyplot as plt

def compute_annotation_stats(label_dir):
    annotation_count = 0
    for label_file in Path(label_dir).rglob("*.txt"):
        with open(label_file, "r") as f:
            lines = f.readlines()
            annotation_count += len([l for l in lines if l.strip()])
    return annotation_count

def annotate_bars(ax, bars):
    """Add value labels on top of bars"""
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

def compute_summary_stats(data_dir):
    sets = ["train", "valid", "test"]
    stats = {"split": [], "images": [], "labels": [], "annotations": []}

    for split in sets:
        img_dir = Path(data_dir) / split / "images"
        label_dir = Path(data_dir) / split / "labels"

        num_images = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
        num_labels = len(list(label_dir.glob("*.txt")))
        total_annotations = compute_annotation_stats(label_dir)

        stats["split"].append(split)
        stats["images"].append(num_images)
        stats["labels"].append(num_labels)
        stats["annotations"].append(total_annotations)

    # Bar chart
    x = range(len(sets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar([p - width for p in x], stats["images"], width=width, label="Images", color="skyblue")
    bars2 = ax.bar(x, stats["labels"], width=width, label="Label Files", color="orange")
    bars3 = ax.bar([p + width for p in x], stats["annotations"], width=width, label="Annotations", color="green")

    annotate_bars(ax, bars1)
    annotate_bars(ax, bars2)
    annotate_bars(ax, bars3)

    ax.set_xticks(x)
    ax.set_xticklabels(stats["split"])
    ax.set_title("Dataset Summary")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()

    # Save plot
    os.makedirs("src/eda/eda_outputs", exist_ok=True)
    plt.savefig("src/eda/eda_outputs/01_summary_statistics.png")
    plt.close()

if __name__ == "__main__":
    compute_summary_stats("data/fc-reali-fscoco-2.v2i.yolov5pytorch")
