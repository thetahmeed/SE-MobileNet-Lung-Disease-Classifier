
## 🛠️ Prerequisites

- **Flutter SDK** (3.0.0 or higher)
- **Android Studio** or **VS Code** with Flutter extension
- **Android SDK** (API level 21 or higher)
- **Physical Android device** or emulator

## 📋 Installation & Setup

### Step 1: Install Flutter

Follow the official Flutter installation guide for your operating system:
- [Flutter Installation Guide](https://docs.flutter.dev/get-started/install)

### Step 2: Verify Flutter Installation

```bash
flutter doctor
```

Ensure all checkmarks are green, especially:
- ✅ Flutter SDK
- ✅ Android toolchain
- ✅ Android Studio / VS Code

### Step 3: Set up the Project

1. **Navigate to the Flutter app directory:**
   ```bash
   cd flutter_app
   ```

2. **Get Flutter dependencies:**
   ```bash
   flutter pub get
   ```

3. **Verify assets are in place:**
   ```bash
   ls -la assets/
   # Should show: covid_classifier.tflite
   ```

### Step 4: Configure Android Permissions

Create or update `android/app/src/main/AndroidManifest.xml`:

```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.example.covid_classifier_app">

    <!-- Internet permission (optional, for debugging) -->
    <uses-permission android:name="android.permission.INTERNET" />
    
    <!-- Camera permissions -->
    <uses-permission android:name="android.permission.CAMERA" />
    
    <!-- Storage permissions for gallery access -->
    <uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
    <uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />

    <application
        android:label="COVID Classifier"
        android:name="${applicationName}"
        android:icon="@mipmap/ic_launcher">
        
        <activity
            android:name=".MainActivity"
            android:exported="true"
            android:launchMode="singleTop"
            android:theme="@style/LaunchTheme"
            android:configChanges="orientation|keyboardHidden|keyboard|screenSize|smallestScreenSize|locale|layoutDirection|fontScale|screenLayout|density|uiMode"
            android:hardwareAccelerated="true"
            android:windowSoftInputMode="adjustResize">
            
            <meta-data
              android:name="io.flutter.embedding.android.NormalTheme"
              android:resource="@style/NormalTheme"
              />
              
            <intent-filter android:autoVerify="true">
                <action android:name="android.intent.action.MAIN"/>
                <category android:name="android.intent.category.LAUNCHER"/>
            </intent-filter>
        </activity>
        
        <meta-data
            android:name="flutterEmbedding"
            android:value="2" />
    </application>
</manifest>
```

### Step 5: Build and Run

1. **Connect your Android device** or start an emulator

2. **Check connected devices:**
   ```bash
   flutter devices
   ```

3. **Build and run the app:**
   ```bash
   # Debug mode (faster for development)
   flutter run
   
   # Release mode (optimized performance)
   flutter run --release
   ```

4. **Build APK for distribution:**
   ```bash
   flutter build apk --release
   # APK will be in: build/app/outputs/flutter-apk/app-release.apk
   ```

## 📁 Project Structure

```
flutter_app/
├── lib/
│   └── main.dart              # Main application code
├── assets/
│   └── covid_classifier.tflite # TensorFlow Lite model
├── android/                   # Android-specific configuration
├── pubspec.yaml              # Flutter dependencies
└── README_FLUTTER.md         # This file
```

## 🔧 Key Dependencies

- **`tflite_flutter`** - TensorFlow Lite integration
- **`image_picker`** - Camera and gallery access
- **`image`** - Image processing and preprocessing
- **`path_provider`** - File system access

## 🎯 How the App Works

### 1. **Model Loading**
- TensorFlow Lite model (`covid_classifier.tflite`) is loaded from assets
- Model size: 2.94 MB (optimized for mobile)
- Input: 224×224×3 RGB images
- Output: 3 probability scores (COVID-19, Normal, Pneumonia)

### 2. **Image Preprocessing**
```dart
// Same preprocessing as Python version:
// 1. Resize to 224×224
// 2. Crop top/bottom 8% (remove text artifacts)
// 3. Normalize pixel values to [0,1]
// 4. Convert to Float32List tensor
```

### 3. **Inference Pipeline**
```dart
// 1. Load image from camera/gallery
// 2. Preprocess image
// 3. Run TensorFlow Lite inference
// 4. Apply softmax to get probabilities
// 5. Display results with confidence scores
```

## 📱 App Interface

### Main Screen Features:
- **Medical Disclaimer** - Important safety notice
- **Model Status** - Shows if TensorFlow Lite model is loaded
- **Image Upload** - Camera capture or gallery selection
- **Analysis Button** - Runs inference on selected image
- **Results Display** - Prediction with confidence scores and detailed probabilities

### User Experience:
1. **📸 Capture/Select** - Take photo or choose from gallery
2. **🔍 Analyze** - Tap analyze button to run classification
3. **📊 Results** - View prediction with confidence percentages
4. **🔄 Repeat** - Analyze multiple images seamlessly

## ⚡ Performance Optimizations

- **Model Quantization** - TensorFlow Lite optimized model
- **Efficient Image Processing** - Native Dart image manipulation
- **Memory Management** - Proper disposal of resources
- **Async Operations** - Non-blocking UI during inference

## 🔒 Privacy & Security

- **Offline Operation** - No data leaves your device
- **Local Processing** - All inference happens on-device
- **No Cloud Dependencies** - Complete privacy protection
- **Temporary Storage** - Images not permanently saved

## 🛠️ Troubleshooting

### Common Issues:

#### 1. **Flutter Doctor Issues**
```bash
flutter doctor --android-licenses  # Accept Android licenses
flutter clean && flutter pub get   # Clean and reinstall dependencies
```

#### 2. **Build Errors**
```bash
cd android && ./gradlew clean      # Clean Android build
cd .. && flutter clean             # Clean Flutter build
flutter pub get                    # Reinstall dependencies
```

#### 3. **TensorFlow Lite Model Not Found**
```bash
# Verify model is in assets/
ls -la assets/covid_classifier.tflite

# Check pubspec.yaml assets section
flutter pub get
```

#### 4. **Camera/Gallery Permissions**
- Ensure permissions are added to AndroidManifest.xml
- Grant permissions manually in device settings if needed

#### 5. **Performance Issues**
```bash
# Use release mode for better performance
flutter run --release

# Or build optimized APK
flutter build apk --release
```

## 📊 Model Information

- **Original Model**: 32.36 MB (.keras)
- **Optimized Model**: 2.94 MB (.tflite)
- **Size Reduction**: 90.9%
- **Architecture**: MobileNetV2
- **Input Size**: 224×224×3
- **Classes**: COVID-19, Normal, Pneumonia
- **Preprocessing**: Normalization, cropping, resizing

## 🚀 Distribution

### Create Release APK:
```bash
flutter build apk --release
```

### APK Location:
```
build/app/outputs/flutter-apk/app-release.apk
```

### Installation:
```bash
# Install on connected device
adb install build/app/outputs/flutter-apk/app-release.apk
```
