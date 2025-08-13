# COVID-19 Chest X-Ray Classifier - Flutter Android App

A native Android application built with Flutter for COVID-19 chest X-ray classification using TensorFlow Lite. This app runs completely offline on your Android device.

## 📁 Project Structure

```
thesis_ui_android/
├── lib/
│   └── main.dart                     # Main application code
├── assets/
│   └── covid_classifier.tflite      # TensorFlow Lite model (2.94 MB)
├── android/                          # Android-specific configuration
│   ├── app/
│   │   ├── build.gradle.kts         # Android build configuration
│   │   ├── proguard-rules.pro       # ProGuard rules for TensorFlow Lite
│   │   └── src/main/AndroidManifest.xml
│   └── build.gradle.kts
├── ios/                              # iOS configuration (if needed)
├── pubspec.yaml                      # Flutter dependencies
└── README.md                         # This file
```

## 🛠️ Prerequisites

- **Flutter SDK** (3.0.0 or higher)
- **Android Studio** or **VS Code** with Flutter extension
- **Android SDK** (API level 21 or higher)
- **Physical Android device** or emulator

## 🚀 How to Run

### Step 1: Verify Flutter Installation

```bash
flutter doctor
```

Ensure all checkmarks are green, especially:

- ✅ Flutter SDK
- ✅ Android toolchain
- ✅ Android Studio / VS Code

### Step 2: Install Dependencies

```bash
# Navigate to project directory
cd thesis_ui_android

# Get Flutter dependencies
flutter pub get
```

### Step 3: Verify Assets

```bash
# Check if TensorFlow Lite model exists
ls -la assets/covid_classifier.tflite
# Should show file with size ~3MB
```

### Step 4: Connect Device

```bash
# Check connected Android devices/emulators
flutter devices
```

### Step 5: Run the App

```bash
# Debug mode (faster for development)
flutter run

# Release mode (optimized performance)
flutter run --release
```

### Step 6: Build APK (Optional)

```bash
# Build APK for distribution
flutter build apk --release

# APK location: build/app/outputs/flutter-apk/app-release.apk
```

## 🔧 Key Dependencies

```yaml
dependencies:
  flutter:
    sdk: flutter
  tflite_flutter: ^0.10.4 # TensorFlow Lite integration
  image_picker: ^1.0.7 # Camera and gallery access
  image: ^4.1.7 # Image processing
  path_provider: ^2.1.2 # File system access
```

## 🔍 Troubleshooting

### Common Issues:

1. **Asset Loading Error**

   ```bash
   # Verify model file exists
   ls -la assets/covid_classifier.tflite

   # Clean and rebuild
   flutter clean
   flutter pub get
   ```

2. **Build Errors**

   ```bash
   # Check Flutter setup
   flutter doctor

   # Clean build files
   flutter clean
   ```

3. **Device Not Detected**

   ```bash
   # Check connected devices
   flutter devices

   # Enable USB debugging on Android device
   ```

### Requirements:

- **Android 5.0** (API level 21) or higher
- **Minimum 2 GB RAM** for optimal performance
- **Camera permission** for capturing images

## 📱 App Features

- **📸 Image Capture** - Camera or gallery selection
- **🔍 Real-time Classification** - COVID-19, Normal, Pneumonia detection
- **📊 Confidence Scores** - Detailed probability results
- **🚀 Offline Operation** - No internet connection required

---

**Built with Flutter & TensorFlow Lite**
