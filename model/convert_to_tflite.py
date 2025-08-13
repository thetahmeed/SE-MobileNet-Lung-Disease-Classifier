#!/usr/bin/env python3
import tensorflow as tf
import numpy as np

def convert_keras_to_tflite():
    """Convert the trained Keras model to TensorFlow Lite format"""
    
    # Load the trained model
    model_path = 'PATH OF .keras file'
    print(f"Loading model from: {model_path}")
    
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    # Convert to TensorFlow Lite
    print("\nConverting to TensorFlow Lite...")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimization options
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Optional: Use float16 quantization for smaller model size
    # converter.target_spec.supported_types = [tf.float16]
    
    try:
        tflite_model = converter.convert()
        print("Model converted successfully!")
    except Exception as e:
        print(f"Error converting model: {e}")
        return False
    
    # Save the TensorFlow Lite model
    tflite_path = 'weight/covid_classifier.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TensorFlow Lite model saved to: {tflite_path}")
    
    # Get model size information
    import os
    original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)  # MB
    
    print(f"\nModel Size Comparison:")
    print(f"   Original (.keras): {original_size:.2f} MB")
    print(f"   TensorFlow Lite (.tflite): {tflite_size:.2f} MB")
    print(f"   Size reduction: {((original_size - tflite_size) / original_size * 100):.1f}%")
    
    # Test the TensorFlow Lite model
    print("\nTesting TensorFlow Lite model...")
    test_tflite_model(tflite_path)
    
    return True

def test_tflite_model(tflite_path):
    """Test the converted TensorFlow Lite model"""
    
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Output shape: {output_details[0]['shape']}")
    
    # Create a random test input
    input_shape = input_details[0]['shape']
    test_input = np.random.random(input_shape).astype(np.float32)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    
    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"   Test inference successful!")
    print(f"   Output shape: {output_data.shape}")
    print(f"TensorFlow Lite model is working correctly!")

if __name__ == "__main__":
    print("COVID-19 Model Converter: Keras to TensorFlow Lite")
    print("=" * 60)
    
    success = convert_keras_to_tflite()
    
    if success:
        print("\nConversion completed successfully!")
    else:
        print("\nConversion failed. Please check the error messages above.") 