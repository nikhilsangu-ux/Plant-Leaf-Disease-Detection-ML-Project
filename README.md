# Plant Leaf Disease Detection using MobileNetV2

An end-to-end deep-learning project that classifies plant leaf images into 38 PlantVillage crop/disease categories using transfer learning with MobileNetV2. Includes dataset preparation, training, evaluation, command-line prediction, and an optional Streamlit interface.

> **Important:** This is an educational prototype, not a definitive agricultural diagnosis. PlantVillage images are mostly collected in controlled conditions, so real farm photos can produce less reliable predictions.

## Features

- PlantVillage dataset downloader using TensorFlow Datasets (avoids the currently inconsistent Hugging Face `color` config issue)
- 80/20 train/test export; validation split made from the training directory
- MobileNetV2 transfer learning with ImageNet weights and data augmentation
- Saved model, class names, training curves, test report, and confusion matrix
- Command-line image prediction and optional Streamlit web app

## Project structure

```text
Plant-Leaf-Disease-Detection/
├── src/
│   ├── check_setup.py
│   ├── download_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── app.py
├── dataset/                 # created on first download; not committed to Git
├── models/                  # trained model and class map
├── results/                 # metrics and plots
├── sample_images/           # put your own leaf image here
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- macOS, Windows, or Linux
- Python 3.11 recommended
- Internet connection for dataset and pretrained model weights
- Several GB of free disk space; exported JPEGs need additional space beyond the compressed dataset

Check your Mac architecture and Python version:

```bash
uname -m
python3 --version
```

## 1. Create the environment

Run these commands from the project root folder in VS Code Terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then check the packages:

```bash
python src/check_setup.py
```

If TensorFlow installation fails, use Python 3.11 in a fresh virtual environment and follow the official TensorFlow install guidance for your Mac architecture. Do not mix packages from different environments.

## 2. Download PlantVillage

```bash
python src/download_dataset.py
```

The first run downloads the PlantVillage dataset through TensorFlow Datasets and exports images into `dataset/PlantVillage/train/` and `dataset/PlantVillage/test/`, grouped into class folders. This may take several minutes and uses substantial disk space. TensorFlow Datasets caches its downloaded files, so a retry may reuse cached data.

The downloader makes a deterministic 80/20 split from the TFDS `train` split. The training script then takes 20% of the training portion for validation.

## 3. Train MobileNetV2

```bash
python src/train.py
```

The initial training phase freezes the pretrained MobileNetV2 feature extractor and trains a new classification head. Training time depends on your Mac; you can adjust `EPOCHS` and `BATCH_SIZE` near the top of `src/train.py` if needed.

Outputs:

- `models/plant_disease_model.keras`
- `models/class_names.json`
- `results/training_log.csv`
- `results/training_curves.png`

## 4. Evaluate the model

```bash
python src/evaluate.py
```

Outputs:

- `results/evaluation_report.txt` (accuracy, precision, recall, F1-score)
- `results/confusion_matrix.png`

Report the measured results from your own run; do not claim a specific accuracy before training and evaluation have completed.

## 5. Predict one image

Copy a leaf photo into `sample_images/`, for example `sample_images/leaf.jpg`, then run:

```bash
python src/predict.py sample_images/leaf.jpg
```

Show five candidates instead of three:

```bash
python src/predict.py sample_images/leaf.jpg --top 5
```

## 6. Run the web interface (optional)

```bash
streamlit run src/app.py
```

A local URL will appear in the terminal. Open it in your browser and upload a JPG or PNG leaf photo.

## Common issues

- **`No module named ...`:** activate `.venv` and run `python -m pip install -r requirements.txt`.
- **TensorFlow installation error:** use Python 3.11 and check the official TensorFlow instructions for your macOS architecture.
- **Dataset download fails:** check internet and disk space, then rerun `python src/download_dataset.py`; TFDS normally caches downloaded source data.
- **Model file missing:** download/export data, then run training before evaluation or prediction.
- **Training is slow:** lower `BATCH_SIZE` to `16` and `EPOCHS` to `3` in `src/train.py` for a quick demonstration run.
- **GitHub rejects a large file:** dataset, model, and generated outputs are ignored by `.gitignore`; do not commit the dataset or trained model.

## GitHub upload

Create an empty repository on GitHub, then run from the project root (replace the URL with your repository URL):

```bash
git init
git add .
git commit -m "Plant Leaf Disease Detection using MobileNetV2"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/Plant-Leaf-Disease-Detection.git
git push -u origin main
```

If Git is already initialized, do not repeat `git init` or add a second remote; check with `git status` and `git remote -v` first.

## Dataset credit

PlantVillage dataset introduced by Mohanty, Hughes, and Salathé, “Using Deep Learning for Image-Based Plant Disease Detection,” *Frontiers in Plant Science* (2016), DOI: 10.3389/fpls.2016.01419. TensorFlow Datasets provides a republished version of the dataset. Check the source dataset's license and citation requirements before redistribution.
