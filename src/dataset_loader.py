from pathlib import Path

from config import CLASSES, DATASET_PATH, NUM_IMAGES_PER_CLASS


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_image_paths():
    dataset_root = Path(DATASET_PATH)
    image_label_pairs = []
    class_counts = {}

    for class_name in CLASSES:
        class_dir = dataset_root / class_name
        if not class_dir.exists():
            class_counts[class_name] = 0
            continue

        image_paths = sorted(
            path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        class_counts[class_name] = len(image_paths)
        image_label_pairs.extend((str(path), class_name) for path in image_paths)

    print("Dataset summary:")
    for class_name in CLASSES:
        count = class_counts.get(class_name, 0)
        print(f"- {class_name}: {count} images (expected around {NUM_IMAGES_PER_CLASS})")
    print(f"Total images: {len(image_label_pairs)}")

    return image_label_pairs


if __name__ == "__main__":
    load_image_paths()
