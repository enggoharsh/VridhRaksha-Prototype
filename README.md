# VridhRaksha – Fall Detection Prototype

**VridhRaksha** is an AI-powered fall detection system designed to protect elderly individuals using non-intrusive, radar-based sensing. This prototype demonstrates fall classification using deep learning models applied on spectrogram images generated from micro-Doppler radar signals.

---

## 🧠 Project Overview

This Jupyter Notebook contains:
- Radar signal preprocessing using Short-Time Fourier Transform (STFT)
- Spectrogram image generation
- Deep learning model (MobileNetV2) for fall classification
- Evaluation metrics (accuracy, precision, recall, F1-score)

---

## 📂 Dataset Access

Due to GitHub file size limitations, the full dataset used for training and evaluation is not included in this repository.

📎 You can download the dataset from the following Google Drive link:  
🔗 [Dataset Folder (Google Drive)](https://drive.google.com/drive/folders/1LyT5tKSvJidWwDO4OxMdpWJSJclwOowG?usp=sharing)

> Drive has all the spectrograms dataset in form of various human activities like falling, walking etc.

---

## 🛠 Requirements

- Python 3.10+
- TensorFlow / Keras
- NumPy, Matplotlib, OpenCV
- Jupyter Notebook

---

## 🚀 Running the Notebook

1. Clone this repository:
   ```bash
   git clone https://github.com/enggoharsh/VridhRaksha-Prototype.git
