"""Optional Streamlit UI for the trained classifier.
Run: streamlit run src/app.py
"""
from pathlib import Path
import json
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "plant_disease_model.keras"
CLASS_PATH = ROOT / "models" / "class_names.json"

st.set_page_config(page_title="Plant Leaf Disease Detection", page_icon="🌿", layout="centered")
st.title("🌿 Plant Leaf Disease Detection")
st.write("Upload a plant leaf image to see the model's most likely class.")
st.caption("Educational prototype only; predictions are not a definitive agricultural diagnosis.")

if not MODEL_PATH.exists() or not CLASS_PATH.exists():
    st.warning("The trained model is not available yet. Run `python src/train.py` after downloading the dataset.")
    st.stop()

@st.cache_resource
def load_artifacts():
    model = keras.models.load_model(MODEL_PATH)
    with open(CLASS_PATH, encoding="utf-8") as f:
        labels = json.load(f)
    return model, labels

model, class_names = load_artifacts()
upload = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png", "webp"])
if upload is not None:
    image = Image.open(upload).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    resized = image.resize((224, 224))
    array = np.asarray(resized, dtype=np.float32)[None, ...]
    probabilities = model.predict(array, verbose=0)[0]
    indices = np.argsort(probabilities)[::-1][:3]
    st.subheader("Top predictions")
    for index in indices:
        st.write(f"**{class_names[int(index)]}** — {probabilities[index] * 100:.2f}%")
