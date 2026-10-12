"""Download PlantVillage through TensorFlow Datasets and export class folders.

Run from the project root:
    python src/download_dataset.py

This deliberately avoids the old Hugging Face custom loading script, whose
configuration is not currently exposed consistently by the Hub's Parquet view.
"""
from pathlib import Path
import sys

try:
    import tensorflow_datasets as tfds
    from PIL import Image
except ImportError as exc:
    print(f"Missing dependency: {exc}")
    print("Activate .venv and run: python -m pip install -r requirements.txt")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dataset" / "PlantVillage"


def export_split(dataset, split_name, label_names):
    split_dir = OUT / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    print(f"Exporting {split_name} images into {split_dir} ...")

    for index, (image_tensor, label_tensor) in enumerate(tfds.as_numpy(dataset)):
        label_id = int(label_tensor)
        if label_id < 0 or label_id >= len(label_names):
            raise ValueError(f"Invalid label ID {label_id} at image {index}")
        class_name = label_names[label_id].replace("/", "_").replace("\\", "_").strip()
        class_dir = split_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        image = Image.fromarray(image_tensor).convert("RGB")
        image.save(class_dir / f"{index:06d}.jpg", "JPEG", quality=92)
        total += 1
        if total % 1000 == 0:
            print(f"  {split_name}: exported {total} images")

    print(f"Finished {split_name}: {total} images")
    return total


def main():
    if OUT.exists() and any(OUT.iterdir()):
        answer = input(f"{OUT} already contains files. Re-export dataset? [y/N]: ").strip().lower()
        if answer != "y":
            print("Cancelled; existing files were left unchanged.")
            return

    print("Loading PlantVillage with TensorFlow Datasets.")
    print("The first run downloads roughly 800 MB; allow extra disk space for extracted JPGs.")
    try:
        (train_ds, test_ds), info = tfds.load(
            "plant_village",
            split=["train[:80%]", "train[80%:]"],
            as_supervised=True,
            with_info=True,
            shuffle_files=True,
        )
    except Exception as exc:
        print("\nCould not download/load PlantVillage.")
        print("Check internet access, available disk space, and TensorFlow installation.")
        print(f"Original error: {type(exc).__name__}: {exc}")
        sys.exit(1)

    label_names = info.features["label"].names
    print(f"Dataset classes: {len(label_names)}")
    print("Creating a deterministic 80/20 train/test split from the TFDS dataset.")
    train_count = export_split(train_ds, "train", label_names)
    test_count = export_split(test_ds, "test", label_names)
    print(f"\nDataset ready at: {OUT}")
    print(f"Training images: {train_count}; test images: {test_count}")
    print("Next command: python src/train.py")


if __name__ == "__main__":
    main()
