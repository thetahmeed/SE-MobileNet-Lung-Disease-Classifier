import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

void main() {
  runApp(const CovidClassifierApp());
}

class CovidClassifierApp extends StatelessWidget {
  const CovidClassifierApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'X-Ray Classifier',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: const ClassifierScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class ClassifierScreen extends StatefulWidget {
  const ClassifierScreen({super.key});

  @override
  State<ClassifierScreen> createState() => _ClassifierScreenState();
}

class _ClassifierScreenState extends State<ClassifierScreen> {
  final ImagePicker _picker = ImagePicker();
  File? _imageFile;
  List<double>? _predictions;
  bool _isLoading = false;
  bool _modelLoaded = false;

  // TensorFlow Lite interpreter
  Interpreter? _interpreter;

  // Class names (same order as training)
  final List<String> _classNames = ['COVID-19', 'Normal', 'Pneumonia'];

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  @override
  void dispose() {
    _interpreter?.close();
    super.dispose();
  }

  // Load TensorFlow Lite model
  Future<void> _loadModel() async {
    try {
      // First try with default options
      try {
        final options = InterpreterOptions();
        _interpreter = await Interpreter.fromAsset(
          'assets/covid_classifier.tflite',
          options: options,
        );
        setState(() {
          _modelLoaded = true;
        });
        print('✅ Model loaded successfully');
        return;
      } catch (e) {
        print('⚠️ Default model loading failed: $e');
      }

      // Try with different options
      try {
        final options = InterpreterOptions();
        // Add number of threads
        options.threads = 1;
        _interpreter = await Interpreter.fromAsset(
          'assets/covid_classifier.tflite',
          options: options,
        );
        setState(() {
          _modelLoaded = true;
        });
        print('✅ Model loaded successfully with custom options');
        return;
      } catch (e) {
        print('⚠️ Custom options model loading failed: $e');
      }

      // If all methods fail, show a helpful error
      throw Exception(
        'The TensorFlow Lite model appears to be incompatible with this device. '
        'The model may have been created with a newer version of TensorFlow. '
        'Please ensure you have the correct model file.',
      );
    } catch (e) {
      print('❌ Error loading model: $e');
      _showError('Failed to load model: $e');
    }
  }

  // Preprocess image for quantized model (uint8 format)
  Uint8List _preprocessImage(img.Image image) {
    // Resize to 224x224
    img.Image resized = img.copyResize(image, width: 224, height: 224);

    // Crop top and bottom 8% (similar to Python preprocessing)
    int cropHeight = (224 * 0.8).round();
    int offsetY = (224 * 0.1).round();
    img.Image cropped = img.copyCrop(
      resized,
      x: 0,
      y: offsetY,
      width: 224,
      height: cropHeight,
    );

    // Resize back to 224x224
    img.Image processed = img.copyResize(cropped, width: 224, height: 224);

    // Convert to Uint8List (quantized model expects uint8 inputs)
    Uint8List input = Uint8List(1 * 224 * 224 * 3);
    int index = 0;

    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {
        img.Pixel pixel = processed.getPixel(x, y);
        // Keep values in [0, 255] range for uint8
        input[index++] = pixel.r.toInt();
        input[index++] = pixel.g.toInt();
        input[index++] = pixel.b.toInt();
      }
    }

    return input;
  }

  // Run inference
  Future<List<double>> _runInference(File imageFile) async {
    if (_interpreter == null) {
      throw Exception('Model not loaded');
    }

    // Read and decode image
    Uint8List imageBytes = await imageFile.readAsBytes();
    img.Image? image = img.decodeImage(imageBytes);

    if (image == null) {
      throw Exception('Failed to decode image');
    }

    // Preprocess image
    Uint8List input = _preprocessImage(image);

    // Prepare input tensor for quantized model
    var inputTensor = input.reshape([1, 224, 224, 3]);

    // Prepare output tensor for quantized model (uint8 output)
    var outputTensor = Uint8List(1 * 3).reshape([1, 3]);

    // Run inference
    _interpreter!.run(inputTensor, outputTensor);

    // Dequantize output and apply softmax to get probabilities
    List<double> rawOutput = [];
    for (int i = 0; i < outputTensor[0].length; i++) {
      int quantizedValue = outputTensor[0][i] as int;
      // Dequantize: (quantized_value - zero_point) * scale
      // Based on model output: scale=0.07633935, zero_point=170
      double dequantizedValue = (quantizedValue - 170) * 0.07633935;
      rawOutput.add(dequantizedValue);
    }

    return _softmax(rawOutput);
  }

  // Softmax function
  List<double> _softmax(List<double> input) {
    double maxVal = input.reduce(math.max);
    List<double> exp = input.map((x) => math.exp(x - maxVal)).toList();
    double sum = exp.reduce((a, b) => a + b);
    return exp.map((x) => x / sum).toList();
  }

  // Pick image from gallery or camera
  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? pickedFile = await _picker.pickImage(
        source: source,
        maxWidth: 1024,
        maxHeight: 1024,
        imageQuality: 85,
      );

      if (pickedFile != null) {
        setState(() {
          _imageFile = File(pickedFile.path);
          _predictions = null;
        });
      }
    } catch (e) {
      _showError('Failed to pick image: $e');
    }
  }

  // Analyze image
  Future<void> _analyzeImage() async {
    if (_imageFile == null || !_modelLoaded) return;

    setState(() {
      _isLoading = true;
    });

    try {
      List<double> predictions = await _runInference(_imageFile!);
      setState(() {
        _predictions = predictions;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
      });
      _showError('Error analyzing image: $e');
    }
  }

  // Show error dialog
  void _showError(String message) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Error'),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  // Get prediction result
  Map<String, dynamic> _getPredictionResult() {
    if (_predictions == null) return {};

    int maxIndex = 0;
    double maxValue = _predictions![0];

    for (int i = 1; i < _predictions!.length; i++) {
      if (_predictions![i] > maxValue) {
        maxValue = _predictions![i];
        maxIndex = i;
      }
    }

    return {
      'class': _classNames[maxIndex],
      'confidence': maxValue,
      'index': maxIndex,
    };
  }

  // Get result color
  Color _getResultColor(String className) {
    switch (className) {
      case 'Normal':
        return Colors.green;
      case 'COVID-19':
        return Colors.red;
      case 'Pneumonia':
        return Colors.orange;
      default:
        return Colors.grey;
    }
  }

  // Get result icon
  IconData _getResultIcon(String className) {
    switch (className) {
      case 'Normal':
        return Icons.check_circle;
      case 'COVID-19':
        return Icons.coronavirus;
      case 'Pneumonia':
        return Icons.warning;
      default:
        return Icons.help;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('X-Ray Classifier'),
        backgroundColor: Colors.blue[700],
        foregroundColor: Colors.white,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Medical disclaimer
            // Container(
            //   padding: const EdgeInsets.all(12),
            //   decoration: BoxDecoration(
            //     color: Colors.orange[50],
            //     border: Border.all(color: Colors.orange[300]!),
            //     borderRadius: BorderRadius.circular(8),
            //   ),
            //   child: const Text(
            //     '⚠️ Medical Disclaimer: This is for educational purposes only. '
            //     'Always consult healthcare professionals for medical diagnosis.',
            //     style: TextStyle(
            //       color: Colors.orange,
            //       fontWeight: FontWeight.w500,
            //     ),
            //   ),
            // ),
            // const SizedBox(height: 20),

            // Model status
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: _modelLoaded ? Colors.green[50] : Colors.red[50],
                border: Border.all(
                  color: _modelLoaded ? Colors.green[300]! : Colors.red[300]!,
                ),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Row(
                children: [
                  Icon(
                    _modelLoaded ? Icons.check_circle : Icons.error,
                    color: _modelLoaded ? Colors.green : Colors.red,
                  ),
                  const SizedBox(width: 8),
                  Text(
                    _modelLoaded ? 'Model Ready' : 'Model Loading...',
                    style: TextStyle(
                      color: _modelLoaded ? Colors.green[700] : Colors.red[700],
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 20),

            // Image upload section
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  children: [
                    const Text(
                      '📤 Upload Chest X-Ray Image',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 16),

                    // Image display
                    Container(
                      height: 250,
                      width: double.infinity,
                      decoration: BoxDecoration(
                        border: Border.all(color: Colors.grey[300]!),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: _imageFile != null
                          ? ClipRRect(
                              borderRadius: BorderRadius.circular(8),
                              child: Image.file(_imageFile!, fit: BoxFit.cover),
                            )
                          : const Center(
                              child: Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  Icon(
                                    Icons.image,
                                    size: 64,
                                    color: Colors.grey,
                                  ),
                                  SizedBox(height: 8),
                                  Text(
                                    'No image selected',
                                    style: TextStyle(
                                      color: Colors.grey,
                                      fontSize: 16,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                    ),

                    const SizedBox(height: 16),

                    // Action buttons
                    Row(
                      children: [
                        Expanded(
                          child: ElevatedButton.icon(
                            onPressed: () => _pickImage(ImageSource.gallery),
                            icon: const Icon(Icons.photo_library),
                            label: const Text('Gallery'),
                            style: ElevatedButton.styleFrom(
                              padding: const EdgeInsets.symmetric(vertical: 12),
                            ),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: ElevatedButton.icon(
                            onPressed: () => _pickImage(ImageSource.camera),
                            icon: const Icon(Icons.camera_alt),
                            label: const Text('Camera'),
                            style: ElevatedButton.styleFrom(
                              padding: const EdgeInsets.symmetric(vertical: 12),
                            ),
                          ),
                        ),
                      ],
                    ),

                    const SizedBox(height: 12),

                    // Analyze button
                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton.icon(
                        onPressed:
                            _imageFile != null && _modelLoaded && !_isLoading
                            ? _analyzeImage
                            : null,
                        icon: _isLoading
                            ? const SizedBox(
                                width: 20,
                                height: 20,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                  valueColor: AlwaysStoppedAnimation<Color>(
                                    Colors.white,
                                  ),
                                ),
                              )
                            : const Icon(Icons.analytics),
                        label: Text(
                          _isLoading ? 'Analyzing...' : '🔍 Analyze Image',
                        ),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.blue[700],
                          foregroundColor: Colors.white,
                          padding: const EdgeInsets.symmetric(vertical: 16),
                          textStyle: const TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 20),

            // Results section
            if (_predictions != null) ...[
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        '📊 Analysis Results',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 16),

                      // Main prediction
                      Builder(
                        builder: (context) {
                          final result = _getPredictionResult();
                          final className = result['class'] as String;
                          final confidence = result['confidence'] as double;

                          return Container(
                            padding: const EdgeInsets.all(16),
                            decoration: BoxDecoration(
                              color: _getResultColor(
                                className,
                              ).withOpacity(0.1),
                              border: Border.all(
                                color: _getResultColor(className),
                                width: 2,
                              ),
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: Row(
                              children: [
                                Icon(
                                  _getResultIcon(className),
                                  color: _getResultColor(className),
                                  size: 32,
                                ),
                                const SizedBox(width: 12),
                                Expanded(
                                  child: Column(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.start,
                                    children: [
                                      Text(
                                        'Prediction: $className',
                                        style: TextStyle(
                                          fontSize: 18,
                                          fontWeight: FontWeight.bold,
                                          color: _getResultColor(className),
                                        ),
                                      ),
                                      Text(
                                        'Confidence: ${(confidence * 100).toStringAsFixed(1)}%',
                                        style: TextStyle(
                                          fontSize: 16,
                                          color: _getResultColor(className),
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ],
                            ),
                          );
                        },
                      ),

                      const SizedBox(height: 16),

                      // Detailed probabilities
                      const Text(
                        'Detailed Probabilities:',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 8),

                      ...List.generate(_classNames.length, (index) {
                        final probability = _predictions![index];
                        return Padding(
                          padding: const EdgeInsets.symmetric(vertical: 4),
                          child: Row(
                            children: [
                              SizedBox(
                                width: 80,
                                child: Text(
                                  _classNames[index],
                                  style: const TextStyle(
                                    fontWeight: FontWeight.w500,
                                  ),
                                ),
                              ),
                              const SizedBox(width: 8),
                              Expanded(
                                child: LinearProgressIndicator(
                                  value: probability,
                                  backgroundColor: Colors.grey[300],
                                  valueColor: AlwaysStoppedAnimation<Color>(
                                    _getResultColor(_classNames[index]),
                                  ),
                                ),
                              ),
                              const SizedBox(width: 8),
                              SizedBox(
                                width: 50,
                                child: Text(
                                  '${(probability * 100).toStringAsFixed(1)}%',
                                  textAlign: TextAlign.right,
                                  style: const TextStyle(
                                    fontWeight: FontWeight.w500,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        );
                      }),
                    ],
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
