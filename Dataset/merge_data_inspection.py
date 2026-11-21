from pathlib import Path
import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

ROOT = Path("merged_dataset")
LABEL_DIR = ROOT / "train" / "labels"
IMAGE_DIR = ROOT / "train" / "images"

CLASS_NAMES = ["cone", "barrier", "stop_sign"]
COLORS = ["lime", "cyan", "red"]

random.seed(42)

samples = {0: [], 1: [], 2: []}

for label_file in LABEL_DIR.glob("*.txt"):
    lines = label_file.read_text().splitlines()

    for line in lines:
        cls = int(line.split()[0])
        if cls in samples:
            samples[cls].append(label_file)

# Select 3 random images for each class
selected = []

for cls in range(3):
    candidates = list(set(samples[cls]))
    chosen = random.sample(candidates, min(3, len(candidates)))
    selected.extend(chosen)

selected = list(dict.fromkeys(selected))

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.flatten()

for ax, label_file in zip(axes, selected):
    image_path = next(
        (IMAGE_DIR / f"{label_file.stem}{ext}"
         for ext in [".jpg", ".jpeg", ".png", ".webp"]
         if (IMAGE_DIR / f"{label_file.stem}{ext}").exists()),
        None
    )

    if image_path is None:
        ax.set_title("Image not found")
        ax.axis("off")
        continue

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    W, H = image.size

    for line in label_file.read_text().splitlines():
        parts = line.split()
        cls = int(parts[0])
        x, y, w, h = map(float, parts[1:])

        x1 = (x - w/2) * W
        y1 = (y - h/2) * H
        x2 = (x + w/2) * W
        y2 = (y + h/2) * H

        color = COLORS[cls]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, max(0, y1-15)), CLASS_NAMES[cls], fill=color)

    ax.imshow(image)
    ax.set_title(label_file.stem, fontsize=9)
    ax.axis("off")

for ax in axes[len(selected):]:
    ax.axis("off")

plt.tight_layout()
plt.show()
