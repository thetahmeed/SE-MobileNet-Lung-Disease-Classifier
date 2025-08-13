# SE-MobileNet Lung Disease Classifier

An interpretable lightweight Squeeze-and-Excitation block-based deep learning model for lung disease prediction from chest X-ray images. This project implements SE-MobileNetV2 architecture to classify COVID-19, Pneumonia, and Normal conditions with high accuracy and visual explanations.

## 🎯 Key Features

- **High Accuracy**: 92.89% classification accuracy (vs 85.24% baseline MobileNetV2)
- **Lightweight**: Optimized for mobile deployment with 90.9% size reduction
- **Interpretable**: Grad-CAM visualization for model explainability
- **Multi-Platform**: Web interface (Gradio), Android app (Flutter), and Colab integration
- **Offline Capable**: TensorFlow Lite model for edge deployment

## 🏗️ Architecture

- **Base Model**: MobileNetV2 with Squeeze-and-Excitation blocks
- **Input**: 224×224×3 chest X-ray images
- **Output**: 3-class classification (COVID-19, Normal, Pneumonia)
- **Model Size**: 2.94 MB (TensorFlow Lite optimized)

## 📊 Performance

| Model                  | Accuracy | Parameters | Size    |
| ---------------------- | -------- | ---------- | ------- |
| MobileNetV2 (Baseline) | 85.24%   | -          | -       |
| SE-MobileNetV2 (Ours)  | 92.89%   | 2.28M      | 2.94 MB |

## 🚀 Quick Start

### Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/SE-MobileNet-Lung-Disease-Classifier/blob/main/training.ipynb)

### Local Setup

```bash
git clone https://github.com/thetahmeed/SE-MobileNet-Lung-Disease-Classifier.git
cd SE-MobileNet-Lung-Disease-Classifier
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 gradio_app.py
```
