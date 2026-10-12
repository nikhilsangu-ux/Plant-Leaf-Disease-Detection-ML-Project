"""Predict the PlantVillage class for one leaf image."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "plant_disease_model.keras"
CLASS_PATH = ROOT / "models" / "class_names.json"


def main():
    parser = argparse.ArgumentParser(description="Classify a plant leaf image")
    parser.add_argument("image", help="Path to a JPG, PNG, or other PIL-readable image")
    parser.add_argument("--top", type=int, default=3, help="Number of top predictions to display")
    args = parser.parse_args()
    if not MODEL_PATH.exists() or not CLASS_PATH.exists():
        print("Trained model not found. Run python src/train.py first.")
        sys.exit(1)
    try:
        image = Image.open(args.image).convert("RGB").resize((224, 224))
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        print(f"Could not open image '{args.image}': {exc}")
        sys.exit(1)
    with open(CLASS_PATH, encoding="utf-8") as f:
        class_names = json.load(f)
    model = keras.models.load_model(MODEL_PATH)
    array = np.asarray(image, dtype=np.float32)[None, ...]
    probabilities = model.predict(array, verbose=0)[0]
    top_n = max(1, min(args.top, len(class_names)))
    indices = np.argsort(probabilities)[::-1][:top_n]
    print("\nPredictions (model confidence, not a guaranteed diagnosis):")
    for index in indices:
        print(f"  {class_names[int(index)]}: {probabilities[index] * 100:.2f}%")
    print("\nNote: Results on real-world farm photos may be less reliable than on PlantVillage images.")


if __name__ == "__main__":
    main()
