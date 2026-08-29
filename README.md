# 🌿 Plant Leaf Disease Detection

A Machine Learning project that detects plant leaf diseases from images using **Deep Learning and Transfer Learning**. The system analyzes a leaf image and predicts whether the leaf is healthy or affected by a particular disease.

## 📌 Project Overview

Plant diseases can significantly affect crop production and farmer income. Early identification of diseases can help farmers take appropriate action at the right time.

This project uses the **PlantVillage dataset** and a **MobileNetV2** deep learning model to classify plant leaf images into different disease categories.

The project can be integrated into a mobile or web application where farmers can upload or capture a photo of a plant leaf and receive a disease prediction.

## 🎯 Objectives

* Detect plant diseases from leaf images.
* Classify leaves as healthy or diseased.
* Identify the specific disease affecting the plant.
* Use Transfer Learning to achieve effective image classification.
* Provide a foundation for an AI-based agricultural assistance system.

## 🧠 Machine Learning Approach

The project uses **Transfer Learning** with **MobileNetV2**.

### Workflow

```text
Plant Leaf Image
       ↓
Image Preprocessing
       ↓
Data Augmentation
       ↓
MobileNetV2
       ↓
Transfer Learning
       ↓
Disease Classification
       ↓
Predicted Disease
```

## 📊 Dataset

### PlantVillage Dataset

The model is trained using the **PlantVillage dataset**, which contains labeled images of healthy and diseased plant leaves.

The dataset includes multiple crops and disease categories.

> Dataset: PlantVillage

## 🛠️ Technologies Used

* **Python**
* **TensorFlow / Keras**
* **MobileNetV2**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Scikit-learn**
* **OpenCV / PIL**
* **Jupyter Notebook**

## 📁 Project Structure

```text
Plant-Leaf-Disease-Detection/
│
├── dataset/
│   └── plantvillage/
│
├── notebooks/
│   └── disease_detection.ipynb
│
├── src/
│   ├── train.py
│   ├── predict.py
│   └── preprocess.py
│
├── models/
│   └── plant_disease_model.keras
│
├── results/
│   ├── accuracy.png
│   └── confusion_matrix.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Plant-Leaf-Disease-Detection.git
```

### 2. Navigate to the project

```bash
cd Plant-Leaf-Disease-Detection
```

### 3. Create a virtual environment

```bash
python3 -m venv venv
```

### 4. Activate the virtual environment

For macOS/Linux:

```bash
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

## 🚀 How to Run

### Train the model

```bash
python src/train.py
```

The trained model will be saved inside the `models/` directory.

### Predict a disease

```bash
python src/predict.py
```

The program will process the leaf image and return the predicted disease class.

## 📈 Model

The project uses **MobileNetV2** with Transfer Learning.

### Model Pipeline

```text
Input Image
    ↓
Resize Image
    ↓
Normalize Pixel Values
    ↓
MobileNetV2 Base Model
    ↓
Global Average Pooling
    ↓
Dense Layer
    ↓
Output Layer
    ↓
Disease Prediction
```

## 🔬 Evaluation

The model can be evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

## 📷 Example Prediction

```text
Input:
Tomato Leaf Image

Prediction:
Crop     : Tomato
Status   : Diseased
Disease  : Tomato Early Blight
```

## 🌾 Real-World Application

This project can be integrated into an agricultural application such as **KissanConnect**.

A possible workflow is:

```text
Farmer
   ↓
Upload / Capture Leaf Photo
   ↓
AI Disease Detection
   ↓
Disease Identification
   ↓
Display Result
   ↓
Provide Agricultural Guidance
```

## ⚠️ Limitations

* Model performance depends on image quality.
* Predictions may be less accurate for leaves photographed in environments different from the training dataset.
* The model should be considered an assistance tool rather than a replacement for professional agricultural diagnosis.
* More real-world field images can be added to improve generalization.

## 🔮 Future Enhancements

* 📱 Develop a mobile application.
* 🌐 Deploy the model as a web API.
* 🌱 Add more crops and diseases.
* 📍 Use location information for agricultural recommendations.
* 🌦️ Combine disease detection with weather information.
* 🤖 Add AI-based agricultural recommendations.
* 🗣️ Support multiple languages such as English, Telugu, and Hindi.
* ☁️ Deploy the model to a cloud platform.

## 👩‍💻 Author

**Nikhil Sangu**

Machine Learning / Deep Learning Project
