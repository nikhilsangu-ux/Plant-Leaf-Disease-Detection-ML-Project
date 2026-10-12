"""Evaluate the saved model on the held-out test directory."""
from pathlib import Path
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "plant_disease_model.keras"
CLASS_PATH = ROOT / "models" / "class_names.json"
TEST_DIR = ROOT / "dataset" / "PlantVillage" / "test"
RESULTS = ROOT / "results"


def main():
    for path in (MODEL_PATH, CLASS_PATH, TEST_DIR):
        if not path.exists():
            print(f"Required path not found: {path}")
            print("Run dataset download and training first.")
            sys.exit(1)
    with open(CLASS_PATH, encoding="utf-8") as f:
        class_names = json.load(f)
    test_ds = keras.utils.image_dataset_from_directory(
        TEST_DIR, image_size=(224, 224), batch_size=32, shuffle=False,
        label_mode="int", class_names=class_names,
    ).prefetch(tf.data.AUTOTUNE)
    model = keras.models.load_model(MODEL_PATH)
    y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
    probabilities = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    RESULTS.mkdir(exist_ok=True)
    report_text = f"Test accuracy: {accuracy:.4f} ({accuracy:.2%})\n\n{report}"
    (RESULTS / "evaluation_report.txt").write_text(report_text, encoding="utf-8")
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(16, 14))
    image = ax.imshow(matrix, interpolation="nearest", aspect="auto")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("PlantVillage Test Confusion Matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(range(len(class_names)), class_names, rotation=90, fontsize=6)
    ax.set_yticks(range(len(class_names)), class_names, fontsize=6)
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    print(report_text)
    print(f"\nSaved report: {RESULTS / 'evaluation_report.txt'}")
    print(f"Saved confusion matrix: {RESULTS / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
