# 🔧 Flutter App Asset Loading Error - Troubleshooting Guide

## 🚨 Error: "Unable to load asset: assets/covid_classifier.tflite"

This error occurs when Flutter can't find the TensorFlow Lite model file. Here's how to fix it:

## ✅ **Step-by-Step Solution**

### **1. Verify File Structure**
Your Flutter app should look like this:
```
flutter_app/
├── lib/
│   └── main.dart
├── assets/
│   └── covid_classifier.tflite  ← This file MUST exist
├── pubspec.yaml
└── android/ (if building for Android)
```

### **2. Check if Model File Exists**
```bash
cd flutter_app
ls -la assets/covid_classifier.tflite
```
**Expected output:** Should show the file with size ~3MB
**If missing:** Copy the model file:
```bash
cp ../weight/covid_classifier.tflite assets/
```

### **3. Verify pubspec.yaml Configuration**
Open `flutter_app/pubspec.yaml` and ensure it contains:
```yaml
flutter:
  uses-material-design: true
  
  assets:
    - assets/covid_classifier.tflite
    - assets/
```

### **4. Clean and Rebuild**
```bash
# Navigate to flutter app directory
cd flutter_app

# Clean previous builds
flutter clean

# Get dependencies
flutter pub get

# Build and run
flutter run --release
```

### **5. Alternative Asset Paths**
If the issue persists, the app tries multiple paths:
- `assets/covid_classifier.tflite` (primary)
- `covid_classifier.tflite` (fallback)
- `assets/models/covid_classifier.tflite` (alternate)

You can also try placing the model in the root:
```bash
cp assets/covid_classifier.tflite covid_classifier.tflite
```

## 🔍 **Debugging Steps**

### **Check Flutter Doctor**
```bash
flutter doctor
```
Ensure all checkmarks are green.

### **Check Connected Devices**
```bash
flutter devices
```
Ensure your Android device/emulator is connected.

### **Enable Verbose Logging**
```bash
flutter run --release --verbose
```
Look for asset-related error messages.

### **Check Build Output**
During build, you should see:
```
✓ Built build/app/outputs/flutter-apk/app-release.apk
```

## 📱 **Platform-Specific Issues**

### **Android Issues**
1. **Permissions**: Ensure `android/app/src/main/AndroidManifest.xml` has:
```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
```

2. **Min SDK**: Check `android/app/build.gradle`:
```gradle
android {
    compileSdkVersion 34
    defaultConfig {
        minSdkVersion 21  // Minimum for TensorFlow Lite
    }
}
```

### **Asset Bundle Issues**
If assets aren't being bundled:
```bash
# Check if assets are included in APK
unzip -l build/app/outputs/flutter-apk/app-release.apk | grep covid_classifier
```

## 🛠️ **Alternative Solutions**

### **Option 1: Use Different Asset Path**
Modify `lib/main.dart` to use a different path:
```dart
// In _loadModel() function, try:
_interpreter = await Interpreter.fromAsset('covid_classifier.tflite');
```

### **Option 2: Load from File System**
```dart
// Copy model to documents directory first
import 'package:path_provider/path_provider.dart';

Future<String> _copyAssetToFile() async {
  final dir = await getApplicationDocumentsDirectory();
  final file = File('${dir.path}/covid_classifier.tflite');
  
  if (!await file.exists()) {
    final data = await rootBundle.load('assets/covid_classifier.tflite');
    await file.writeAsBytes(data.buffer.asUint8List());
  }
  
  return file.path;
}

// Then load from file path
final modelPath = await _copyAssetToFile();
_interpreter = await Interpreter.fromFile(File(modelPath));
```

### **Option 3: Verify Model in APK**
```bash
# Extract and check APK contents
cd build/app/outputs/flutter-apk/
unzip app-release.apk
ls -la flutter_assets/assets/
```

## 🔄 **Complete Reset Process**

If nothing works, try a complete reset:

```bash
# 1. Clean everything
flutter clean
rm -rf build/
rm -rf .dart_tool/

# 2. Verify model exists
ls -la assets/covid_classifier.tflite

# 3. Check pubspec.yaml
cat pubspec.yaml | grep -A 5 "assets:"

# 4. Get dependencies
flutter pub get

# 5. Build step by step
flutter build apk --debug --verbose

# 6. Check for errors in output
flutter run --release
```

## 📋 **Quick Checklist**

- [ ] Model file exists in `assets/covid_classifier.tflite`
- [ ] File size is ~3MB (3,084,400 bytes)
- [ ] `pubspec.yaml` lists the asset correctly
- [ ] Ran `flutter pub get`
- [ ] Ran `flutter clean`
- [ ] No errors in `flutter doctor`
- [ ] Device/emulator is connected
- [ ] Using `flutter run --release` for better performance

## 💡 **Pro Tips**

1. **Always use release mode** for testing ML models: `flutter run --release`
2. **Check file permissions** on the model file: `chmod 644 assets/covid_classifier.tflite`
3. **Verify model integrity** by checking file size matches original
4. **Use Android Studio** for better debugging and error messages
5. **Test on physical device** rather than emulator for better performance

## 🚨 **If Still Not Working**

Try this minimal test to isolate the issue:

```dart
// Add to _loadModel() function for debugging:
try {
  final byteData = await rootBundle.load('assets/covid_classifier.tflite');
  print('✅ Asset loaded successfully, size: ${byteData.lengthInBytes} bytes');
} catch (e) {
  print('❌ Asset loading failed: $e');
}
```

This will help determine if it's an asset loading issue or TensorFlow Lite issue specifically.

---

**Need more help?** Check the Flutter console output for specific error messages and follow the debugging steps above. The app includes built-in troubleshooting UI and retry functionality. 