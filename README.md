# An interpretable lightweight squeeze-and-excitation block-based deep learning model for lung disease prediction

Abstract—The term “lung disease” refers to any condition
affecting the lungs that impairs their ability to function properly.
Many disorders, such as asthma, COPD, infections like influenza,
pneumonia, tuberculosis, lung cancer, and many other breathing
problems, are common types of lung diseases around the world.
Early detection and diagnosis are crucial for lung disease since it
can cause loss of life. Along with the conventional clinical labora-
tory methods, some DL models are widely used as computer-aided
detection (CAD) for medical purposes to predict and classify
diseases or any abnormalities. So we want to develop such a
deep learning (DL) model, which will be more lightweight and
less costly but more effective than any other existing DL models.
We proposed to integrate the SE block into MobileNet to form
the SE-MobileNet module-based model because, unlike the other
basic CNNs that treat all feature channels equally, the SE block
focuses on the most important features, and MobileNet works
well on mobile devices with less power, as it is a lightweight
version of CNN. By highlighting affected areas in lung images,
this combination results in a simple, lightweight, smarter model
to boost early lung disease prediction performance, which can be
very useful in the healthcare field. We ran MobileNetV2 and our
proposed SE-MobileNetV2 with accuracy values of 85.24% and
92.89%, respectively. We also use Grad-CAM and then develop
it on a mobile app that gives real-time visual explanations of
our model’s predictions. We are hopeful about our model, which
may be an optimal solution to lung disease detection at early
stages and will make a benchmark contribution to the medical
industry.

Keywords — Lung Disease, Squeeze-and-Excitation Block,
Lightweight Deep Learning, MobileNet, Explainable AI

## Key Features
- **High Accuracy**: 92.89% classification accuracy (vs 85.24% baseline MobileNetV2)
- **Lightweight**: Optimized for mobile deployment with 90.9% size reduction
- **Interpretable**: Grad-CAM visualization for model explainability
- **Multi-Platform**: Web interface (Gradio), Android app (Flutter), and Colab integration
- **Offline Capable**: TensorFlow Lite model for edge deployment

## Architecture
- **Base Model**: MobileNetV2 with Squeeze-and-Excitation blocks
- **Input**: 224×224×3 chest X-ray images
- **Output**: 3-class classification (COVID-19, Normal, Pneumonia)
- **Model Size**: 2.94 MB (TensorFlow Lite optimized)

## Methodology

![Methodology](https://res.cloudinary.com/danjv8onb/image/upload/v1755090064/Screenshot_2025-08-13_at_7.00.55_PM_trqsbh.png)

## Links
- **Google Colab**: [Open](https://colab.research.google.com/github/thetahmeed/SE-MobileNet-Lung-Disease-Classifier/blob/main/model/main.ipynb)
- **Dataset**: [Download](https://drive.google.com/file/d/1jMJy-Tn2warwOR5BDe3AtoY2HjOU2AUV/view?usp=sharing)
- **Trained Weights**: [Download](https://drive.google.com/drive/folders/1KlZj5lwCYudQVd9Ev4hCsyjQszGJGSNt?usp=sharing)
- **Mobile App (Android)**: [Download](https://drive.google.com/file/d/1a3BBbUdwkgJJTvrIkgJsGG0S90xiwQ3O/view?usp=sharing)

- **Grad-CAM**: [Guide](#2-grad-cam-visualization-modelgrad_campy)
- **Web Interface**: [Guide](#3-web-interface-modelgradio_apppy)



## Project Structure

```
SE-MobileNet-Lung-Disease-Classifier/
├── dataset/
│   ├── dataset.txt          # Google Drive link to dataset
│   └── dataset_ref.txt      # Dataset reference information
├── mobile-app/              # Flutter Android application
│   ├── lib/
│   │   └── main.dart        # Main Flutter application code
│   ├── assets/
│   │   └── covid_classifier.tflite  # TensorFlow Lite model
│   ├── android/             # Android-specific configuration
│   ├── ios/                 # iOS configuration (if needed)
│   ├── pubspec.yaml         # Flutter dependencies
│   ├── FLUTTER.md           # Detailed Flutter setup guide
│   └── FLUTTER_TROUBLESHOOTING.md
├── model/                   # Main model training and inference
│   ├── main.ipynb           # Jupyter notebook for model training
│   ├── grad_cam.py          # Grad-CAM visualization script
│   ├── gradio_app.py        # Web interface using Gradio
│   ├── convert_to_tflite.py # Model conversion script
│   ├── requirements.txt     # Python dependencies
│   ├── test_images/         # Sample test images
│   │   └── readme.txt       # Instructions for test images
│   ├── weight/              # Model weights storage
│   │   └── readme.txt       # Instructions for downloading weights
│   ├── venv/                # Python virtual environment
│   └── README.md            # Detailed model setup instructions
├── output/                  # Model outputs and results
│   ├── results.txt          # Training results
│   └── weight.txt           # Links to pre-trained weights
└── README.md                # This file
```

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/thetahmeed/SE-MobileNet-Lung-Disease-Classifier.git
cd SE-MobileNet-Lung-Disease-Classifier
```

## Model Training & Analysis

### 1. Training the Model (`model/main.ipynb`)

The main Jupyter notebook contains the complete training pipeline for the SE-MobileNetV2 model.

#### Prerequisites

- Google Colab (recommended) or local Jupyter environment
- Google Drive account for dataset storage

#### Setup Instructions

1. **Download the dataset:**
   - Download from: https://drive.google.com/file/d/1jMJy-Tn2warwOR5BDe3AtoY2HjOU2AUV/view?usp=sharing
   - Create a folder named `MyThesis` on Google Drive
   - Upload the dataset.zip file to the `MyThesis` folder
   - Create a `result` folder inside `MyThesis` for storing outputs

2. **Run in Google Colab:**
   ```bash
   # Upload main.ipynb to Google Colab
   # OR open directly: https://colab.research.google.com/github/thetahmeed/SE-MobileNet-Lung-Disease-Classifier/blob/main/model/main.ipynb
   ```

3. **Connect to runtime and execute all cells**

#### What the notebook does:

- Data preprocessing and augmentation
- SE-MobileNetV2 model architecture implementation
- Training with transfer learning
- Model evaluation and performance metrics
- TensorFlow Lite conversion for mobile deployment

### 2. Grad-CAM Visualization (`model/grad_cam.py`)

![Grad-CAM Visualization](https://res.cloudinary.com/danjv8onb/image/upload/v1755089349/grad-cam_k9mvnt.jpg)

Generate visual explanations for model predictions using Gradient-weighted Class Activation Mapping.

#### Setup

1. **Create virtual environment:**
   ```bash
   cd model
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download required files:**
   - **Test Images**: Download 3 sample images from the [dataset link](https://drive.google.com/file/d/1jMJy-Tn2warwOR5BDe3AtoY2HjOU2AUV/view?usp=sharing)
   - Save them in `model/test_images/` with names:
     - `covid.jpg`
     - `normal.png`
     - `pneumonia.jpeg`
   
   - **Model Weights**: Download from [Google Drive](https://drive.google.com/drive/folders/1KlZj5lwCYudQVd9Ev4hCsyjQszGJGSNt?usp=sharing)
   - Save as `model/weight/weight.keras`

4. **Run Grad-CAM analysis:**
   ```bash
   python3 grad_cam.py
   ```

#### Output

- Generates heatmap visualizations showing which regions of the X-ray the model focuses on
- Saves visualization images with overlaid attention maps
- Helps understand model decision-making process

### 3. Web Interface (`model/gradio_app.py`)

Launch an interactive web interface for real-time lung disease classification.

#### Setup

1. **Activate virtual environment:**
   ```bash
   cd model
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download model weights:**
   - Download `weight.keras` from [Google Drive](https://drive.google.com/drive/folders/1KlZj5lwCYudQVd9Ev4hCsyjQszGJGSNt?usp=sharing)
   - Place in `model/weight/weight.keras`

4. **Launch the web app:**
   ```bash
   python3 gradio_app.py
   ```

5. **Access the interface:**
   - Open browser and go to: `http://localhost:7860`
   - Or use the public URL provided in the terminal

#### Features

- **Image Upload**: Drag and drop or click to upload chest X-ray images
- **Real-time Classification**: Instant prediction results
- **Confidence Scores**: Probability distribution for all classes
- **User-friendly Interface**: Clean, intuitive web interface
- **Shareable**: Generate public links for demonstrations

## Mobile App Setup & Usage

![Mobile App Screenshot](https://res.cloudinary.com/danjv8onb/image/upload/v1755089275/mobile-app-ss_1_obj2vp.png)

The project includes a Flutter Android application for real-time lung disease classification.

### Prerequisites

- **Flutter SDK** (3.8.1 or higher)
- **Android Studio** or **VS Code** with Flutter extension
- **Android SDK** (API level 21 or higher)
- **Physical Android device** or emulator

### Installation Steps

1. **Install Flutter** following the [official guide](https://docs.flutter.dev/get-started/install)

2. **Verify Flutter installation:**
   ```bash
   flutter doctor
   ```

3. **Navigate to the mobile app directory:**
   ```bash
   cd mobile-app
   ```

4. **Get Flutter dependencies:**
   ```bash
   flutter pub get
   ```

5. **Connect your Android device** or start an emulator

6. **Check connected devices:**
   ```bash
   flutter devices
   ```

7. **Run the app:**
   ```bash
   # Debug mode (faster for development)
   flutter run
   
   # Release mode (optimized performance)
   flutter run --release
   ```

8. **Build APK for distribution:**
   ```bash
   flutter build apk --release
   # APK will be in: build/app/outputs/flutter-apk/app-release.apk
   ```

### App Features

- **Camera Integration**: Capture chest X-ray images directly
- **Gallery Support**: Select images from device gallery
- **Real-time Analysis**: Instant classification using TensorFlow Lite
- **Detailed Results**: Confidence scores and probability visualization
- **Offline Operation**: No internet connection required

For detailed setup instructions, see [`mobile-app/FLUTTER.md`](mobile-app/FLUTTER.md).

## Dependencies

### Python Requirements (`model/requirements.txt`)

```
tf-nightly
gradio>=5.0.0
numpy>=1.21.0
matplotlib>=3.5.0
pillow>=8.0.0
opencv-python>=4.5.0
```

### Flutter Dependencies (`mobile-app/pubspec.yaml`)

```yaml
dependencies:
  flutter:
    sdk: flutter
  cupertino_icons: ^1.0.8
  tflite_flutter: ^0.11.0
  image_picker: ^1.0.4
  image: ^4.1.3
  path_provider: ^2.1.1
  flutter_card_swiper: ^7.0.1
```

### Medical Disclaimer
This application is for **educational and research purposes only**. It should **NOT** be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical advice and diagnosis.

### Contributors

**Authors:**
1. **Md. Tahmeedul Islam**  
   ID: 2125702012  
   Department of Computer Science and Engineering  
   City University, Bangladesh

2. **Shifa Somaiya Ritu**  
   ID: 2125702012  
   Department of Computer Science and Engineering  
   City University, Bangladesh

**Supervisor:**  
**Diponkor Bala**, Lecturer  
Department of Computer Science and Engineering  
City University, Bangladesh

**Special Thanks:**  
**Sraboni Ghosh Joya**, Lecturer  
Department of Computer Science and Engineering  
City University, Bangladesh

---
