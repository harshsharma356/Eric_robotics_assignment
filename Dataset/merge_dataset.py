from pathlib import Path
import shutil

ROOT = Path.cwd()
OUT = ROOT / "merged_dataset"

NUIMAGES = ROOT / "Nuimages 5k.v1-main-version--march-24th.yolov11"
SIGNS = ROOT / "US Traffic Signs.v1i.yolov11"

# Original class ID -> merged class ID
MAPPINGS = {
    "nuimages": {
        2: 0,  # traffic cone
        1: 1,  # barrier
    },
    "signs": {
        19: 2,  # stop sign
    },
}

SOURCES = {
    "nuimages": NUIMAGES,
    "signs": SIGNS,
}

SPLITS = ["train", "valid", "test"]
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def find_image(images_dir, stem):
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    # Case-insensitive extension fallback
    matches = [
        p for p in images_dir.iterdir()
        if p.stem == stem and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return matches[0] if matches else None


def process_source(source_name, source_root, mapping):
    total_images = 0
    total_objects = {0: 0, 1: 0, 2: 0}

    for split in SPLITS:
        src_images = source_root / split / "images"
        src_labels = source_root / split / "labels"

        dst_images = OUT / split / "images"
        dst_labels = OUT / split / "labels"

        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        for label_file in src_labels.glob("*.txt"):
            filtered_lines = []

            for line in label_file.read_text().splitlines():
                parts = line.strip().split()

                if len(parts) < 5:
                    continue

                try:
                    original_id = int(parts[0])
                except ValueError:
                    continue

                if original_id not in mapping:
                    continue

                new_id = mapping[original_id]
                parts[0] = str(new_id)
                filtered_lines.append(" ".join(parts))
                total_objects[new_id] += 1

            # Skip images that don't contain our required class(es)
            if not filtered_lines:
                continue

            image_path = find_image(src_images, label_file.stem)

            if image_path is None:
                print(f"WARNING: Image missing for {label_file}")
                continue

            # Prefix prevents filename collisions between datasets
            new_stem = f"{source_name}_{label_file.stem}"

            shutil.copy2(
                image_path,
                dst_images / f"{new_stem}{image_path.suffix.lower()}"
            )

            (dst_labels / f"{new_stem}.txt").write_text(
                "\n".join(filtered_lines) + "\n"
            )

            total_images += 1

        print(f"\n{source_name.upper()} | {split}")
        print(f"Images copied: {total_images}")

    return total_images, total_objects


def main():
    if OUT.exists():
        print(f"ERROR: {OUT} already exists.")
        print("Please rename/delete it before rerunning to avoid mixing datasets.")
        return

    OUT.mkdir(parents=True)

    summary = {}

    for source_name, source_root in SOURCES.items():
        images, objects = process_source(
            source_name,
            source_root,
            MAPPINGS[source_name]
        )
        summary[source_name] = (images, objects)

    # Create merged YOLO data.yaml
    yaml_content = """\
train: train/images
val: valid/images
test: test/images

nc: 3
names:
  0: cone
  1: barrier
  2: stop_sign
"""

    (OUT / "data.yaml").write_text(yaml_content)

    print("\n" + "=" * 50)
    print("MERGE COMPLETE")
    print("=" * 50)

    for split in SPLITS:
        image_count = len(list((OUT / split / "images").glob("*")))
        label_count = len(list((OUT / split / "labels").glob("*.txt")))
        print(f"{split.upper():5s} | Images: {image_count} | Labels: {label_count}")

    print("\nClass totals:")
    for class_id, class_name in enumerate(["cone", "barrier", "stop_sign"]):
        total = sum(
            objects[class_id]
            for _, objects in summary.values()
        )
        print(f"{class_name:10s}: {total} annotations")

    print(f"\nDataset location: {OUT}")


if __name__ == "__main__":
    main()
