"""Train a MobileNetV2 plant disease classifier."""
from pathlib import Path
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "PlantVillage" / "train"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 8


def main():
    if not DATA_DIR.exists() or not any(p.is_dir() for p in DATA_DIR.iterdir()):
        print("Training folders not found. First run: python src/download_dataset.py")
        sys.exit(1)
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    train_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="int",
    )
    val_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="int",
    )
    class_names = train_ds.class_names
    if len(class_names) < 2:
        raise RuntimeError(f"Expected at least 2 classes, found {len(class_names)}")
    with open(MODEL_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")
    base = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = augmentation(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(len(class_names), activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(MODEL_DIR / "plant_disease_model.keras"),
                                        monitor="val_accuracy", save_best_only=True),
        keras.callbacks.CSVLogger(str(RESULTS_DIR / "training_log.csv")),
    ]
    print(f"Training MobileNetV2 on {len(class_names)} classes...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    model.save(MODEL_DIR / "plant_disease_model.keras")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history.get("accuracy", []), label="Train")
    plt.plot(history.history.get("val_accuracy", []), label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history.get("loss", []), label="Train")
    plt.plot(history.history.get("val_loss", []), label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_curves.png", dpi=150)
    plt.close()
    print("Training complete.")
    print(f"Model: {MODEL_DIR / 'plant_disease_model.keras'}")
    print(f"Class labels: {MODEL_DIR / 'class_names.json'}")
    print(f"Curves: {RESULTS_DIR / 'training_curves.png'}")
    print("Next command: python src/evaluate.py")


if __name__ == "__main__":
    main()
