# 🌿 Plant Leaf Disease Detection Using Machine Learning

A deep learning-based application that identifies plant leaf diseases from images using **MobileNetV2 Transfer Learning**, TensorFlow, and Streamlit. The application allows users to upload a leaf image and receive predicted disease classes along with confidence scores.

## 📌 Project Overview

Plant diseases can significantly affect crop quality and agricultural productivity. Early identification can help farmers and agricultural researchers take appropriate action.

This project uses a convolutional neural network based on MobileNetV2, pretrained on ImageNet and fine-tuned for plant disease classification. The trained model predicts the most likely disease category from an uploaded leaf image.

The model was evaluated on a held-out test dataset and achieved **95.35% test accuracy** in the recorded evaluation.

## ✨ Features

- 🌱 Plant leaf disease classification from images
- 🧠 MobileNetV2 transfer learning
- 📊 Top-three predicted classes with confidence scores
- 🖥️ Interactive Streamlit web interface
- 📈 Training curves and evaluation reports
- 🔍 Confusion matrix visualization
- 🧪 Single-image prediction through a command-line script
- ♻️ Reusable model training and evaluation scripts

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.11 | Programming language |
| TensorFlow / Keras | Deep learning and model training |
| MobileNetV2 | Transfer learning architecture |
| Streamlit | Interactive web application |
| NumPy | Numerical operations |
| Matplotlib | Training curves and visualizations |
| Scikit-learn | Evaluation metrics |
| PlantVillage | Plant leaf disease image dataset |
| Git and GitHub | Version control and project hosting |

## 📂 Project Structure

```text
Plant-Leaf-Disease-Detection/
│
├── dataset/                  # Dataset files (not tracked by Git)
│
├── models/
│   ├── plant_disease_model.keras
│   └── class_names.json
│
├── results/
│   ├── confusion_matrix.png
│   ├── evaluation_report.txt
│   ├── training_curves.png
│   └── training_log.csv
│
├── sample_images/            # Sample leaf images
│
├── src/
│   ├── app.py                # Streamlit web application
│   ├── check_setup.py        # Environment verification
│   ├── download_dataset.py   # Dataset preparation/download
│   ├── evaluate.py           # Model evaluation
│   ├── predict.py             # Single-image prediction
│   └── train.py              # Model training
│
├── .gitignore
├── README.md
└── requirements.txt
```

## 💻 Hardware Requirements

- Processor: Intel or Apple Silicon processor
- RAM: 8 GB recommended
- Storage: Sufficient free space for the dataset, Python environment, and model files
- GPU: Optional; CPU training and inference may be slower

## 💿 Software Requirements

- macOS, Windows, or Linux
- Python 3.11 recommended
- pip
- Git
- Visual Studio Code or another code editor
- A modern web browser

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/nikhilsangu-ux/Plant-Leaf-Disease-Detection-ML-Project.git
cd Plant-Leaf-Disease-Detection-ML-Project
```

### 2. Create a virtual environment

For macOS and Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

For Windows:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If a dependency fails to install, check that your Python version is compatible with the package versions in `requirements.txt`.

### 4. Prepare the dataset

The complete dataset is not included in this Git repository because it occupies approximately 1.3 GB.

Download or prepare the PlantVillage dataset using the project's dataset preparation script:

```bash
python src/download_dataset.py
```

Follow the script's output and ensure the dataset is stored in the directory expected by `src/train.py` and `src/evaluate.py`.

If the script requires a dataset source or configuration, follow the instructions provided by the script.

### 5. Verify the environment

```bash
python src/check_setup.py
```

## 🧠 Model Training

To train the model from scratch, run:

```bash
python src/train.py
```

The training script uses MobileNetV2 transfer learning to learn features relevant to plant disease classification.

The trained model and class labels are saved in the `models/` directory. Training visualizations and logs are saved in the `results/` directory.

**Note:** Training from scratch requires the dataset and may take time depending on your hardware.

## 📊 Model Evaluation

Evaluate the trained model using:

```bash
python src/evaluate.py
```

The evaluation process calculates classification metrics and generates evaluation artifacts, including:

- Test accuracy
- Precision, recall, and F1-score
- Classification report
- Confusion matrix

The generated results are stored in the `results/` directory.

### Recorded Evaluation Result

| Metric | Result |
|---|---:|
| Test accuracy | 95.35% |
| Macro-average precision | 0.95 |
| Macro-average recall | 0.94 |
| Macro-average F1-score | 0.94 |
| Weighted-average precision | 0.96 |
| Weighted-average recall | 0.95 |
| Weighted-average F1-score | 0.95 |

These values reflect the recorded evaluation run. Results may vary if the dataset, split, preprocessing, or training configuration changes.

## 🌐 Run the Web Application

After installing the dependencies and ensuring the trained model files are available, launch the Streamlit application:

```bash
python -m streamlit run src/app.py
```

Streamlit will display a local URL in the terminal, typically:

```text
http://localhost:8501
```

Open that address in your browser.

### How to Use the Application

1. Launch the application.
2. Upload a plant leaf image using the upload control.
3. Wait for the model to process the image.
4. Review the predicted disease class and confidence score.
5. Compare the top-three predictions if displayed by the application.

**Important:** Model predictions are estimates and should not be treated as a confirmed agricultural diagnosis.

## 🔎 Predict a Single Image

To classify an image from the command line, run:

```bash
python src/predict.py sample_images/leaf.jpg
```

Replace `leaf.jpg` with the actual image filename.

Ensure the image exists at the specified path and that the trained model and class-label files are present in `models/`.

## 📈 Results and Visualizations

The project generates the following outputs:

- `results/training_curves.png` — visualization of training progress
- `results/confusion_matrix.png` — comparison of true and predicted classes
- `results/evaluation_report.txt` — detailed classification metrics
- `results/training_log.csv` — recorded training metrics

These files can be used for project demonstrations, analysis, and documentation.

## 🌾 Dataset Information

This project uses the PlantVillage plant leaf image dataset, which contains images representing multiple plant species and disease categories.

The evaluated model recognizes **38 classes** in the recorded experiment.

The dataset is excluded from Git because of its size. Users must prepare or download the dataset separately before training or evaluation.

## ⚠️ Limitations

- Predictions depend on image quality and similarity to the training data.
- Backgrounds, lighting, leaf orientation, and image artifacts may affect results.
- High confidence does not guarantee that a prediction is correct.
- The model should be evaluated on additional real-world images before practical agricultural deployment.
- Predictions are intended for educational and research purposes, not as a substitute for expert agricultural advice.

## 🔮 Future Enhancements

- Add treatment and prevention recommendations for identified diseases.
- Expand testing with field-collected leaf images.
- Improve robustness to different lighting and backgrounds.
- Deploy the application to a cloud hosting platform.
- Add prediction history and downloadable reports.
- Support additional crops and disease categories.

## 👥 Project Purpose

This project demonstrates how deep learning and transfer learning can be applied to plant disease classification through an accessible web interface.

## 📄 License

Add an appropriate license before redistributing the project. Ensure that the dataset and any pretrained model components are used according to their respective licenses.

## 🙌 Acknowledgements

- TensorFlow and Keras for deep learning tools
- The MobileNetV2 architecture and its pretrained weights
- The PlantVillage dataset for plant leaf image data
- Streamlit for the interactive web application framework
