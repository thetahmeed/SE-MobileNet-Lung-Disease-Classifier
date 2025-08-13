#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Set random seeds for reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Class names from the training notebook
class_names = ['covid', 'normal', 'pneumonia']

def preprocess(image):
    """Preprocess an image applying normalization, cropping and resizing"""
    
    # Convert to float and normalize
    image = tf.cast(image, tf.float32) / 255.0
    
    # Resize to 224x224
    image = tf.image.resize(image, [224, 224])
    
    # Crop top and bottom 8% to remove text information
    image = tf.image.crop_to_bounding_box(
        image, 
        offset_height=int(224*0.1), 
        offset_width=0, 
        target_height=int(224*0.8), 
        target_width=int(224)
    )
    
    # Resize back to 224x224
    image = tf.image.resize(image, [224, 224])
    
    return image

def load_and_preprocess_image(image_path):
    """Load an image file and preprocess it"""
    
    # Read image file
    image = tf.io.read_file(image_path)
    
    # Decode image (supports BMP, GIF, JPEG, PNG)
    image = tf.io.decode_image(image, channels=3)
    
    # Ensure the image has shape [height, width, channels]
    image = tf.ensure_shape(image, [None, None, 3])
    
    # Preprocess the image
    processed_image = preprocess(image)
    
    return processed_image, image

def predict_image(model, image_path):
    """Make prediction on a single image and generate Grad-CAM"""
    
    # Load and preprocess the image
    processed_image, original_image = load_and_preprocess_image(image_path)
    
    # Add batch dimension
    batch_image = tf.expand_dims(processed_image, 0)
    
    # Make prediction
    predictions = model(batch_image)
    
    # Apply softmax to get probabilities
    probabilities = tf.nn.softmax(predictions[0])
    
    # Get predicted class
    predicted_class_idx = tf.argmax(probabilities).numpy()
    predicted_class = class_names[predicted_class_idx]
    confidence = probabilities[predicted_class_idx].numpy()
    
    # Generate Grad-CAM for the predicted class
    try:
        heatmap = generate_gradcam(model, batch_image, predicted_class_idx)
        gradcam_overlay = overlay_gradcam(original_image.numpy(), heatmap)
    except Exception as e:
        print(f"Warning: Could not generate Grad-CAM for {image_path}: {e}")
        heatmap = None
        gradcam_overlay = None
    
    return predicted_class, confidence, probabilities.numpy(), original_image, processed_image, heatmap, gradcam_overlay

def visualize_predictions(results):
    """Visualize the predictions for all images including Grad-CAM heatmaps"""
    
    n_images = len(results)
    # Create 2 rows: processed and grad-cam
    fig, axes = plt.subplots(2, n_images, figsize=(5*n_images, 10))
    
    if n_images == 1:
        axes = axes.reshape(2, 1)
    
    for i, (image_path, predicted_class, confidence, probabilities, original_image, processed_image, heatmap, gradcam_overlay) in enumerate(results):
        # Processed image with predictions
        axes[0, i].imshow(processed_image)
        axes[0, i].set_title(f'Prediction: {predicted_class}\nConfidence: {confidence:.2%}')
        axes[0, i].axis('off')
        
        # Add probability text
        prob_text = '\n'.join([f'{class_names[j]}: {probabilities[j]:.2%}' 
                              for j in range(len(class_names))])
        axes[0, i].text(0.02, 0.98, prob_text, transform=axes[0, i].transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='white', alpha=0.8))
        
        # Grad-CAM overlay
        if gradcam_overlay is not None:
            axes[1, i].imshow(gradcam_overlay)
            axes[1, i].set_title(f'Grad-CAM: Areas of Focus\nfor {predicted_class} prediction')
        else:
            axes[1, i].imshow(original_image)
            axes[1, i].set_title('Grad-CAM: Not Available')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_gradcam(model, image, class_index, layer_name=None):
    """
    Generate Grad-CAM heatmap for a given image and class
    
    Args:
        model: Trained TensorFlow model
        image: Preprocessed image tensor (batch_size, height, width, channels)
        class_index: Index of the class to generate CAM for
        layer_name: Name of the convolutional layer to use (if None, uses last conv layer)
    
    Returns:
        heatmap: Grad-CAM heatmap
    """
    
    # Find the last convolutional layer if not specified
    if layer_name is None:
        conv_layers = []
        for layer in model.layers:
            layer_type = layer.__class__.__name__
            # Look for Conv2D, DepthwiseConv2D, or any convolutional layer
            if any(conv_type in layer_type for conv_type in ['Conv2D', 'DepthwiseConv', 'Conv']):
                conv_layers.append(layer.name)
        
        if conv_layers:
            layer_name = conv_layers[-1]  # Use the last convolutional layer
        else:
            raise ValueError("No convolutional layer found in the model")
    
    print(f"Using layer: {layer_name} for Grad-CAM")
    
    # Create a model that maps the input image to the activations of the target conv layer
    # as well as the output predictions
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs, 
            outputs=[model.get_layer(layer_name).output, model.output]
        )
    except Exception as e:
        raise ValueError(f"Error creating grad model: {e}")
    
    # Compute the gradient of the top predicted class for our input image
    # with respect to the activations of the target conv layer
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        # Handle case where predictions might be a list or have extra dimensions
        if isinstance(predictions, list):
            predictions = predictions[0]
        if len(predictions.shape) > 2:
            predictions = tf.squeeze(predictions)
        loss = predictions[0, class_index]  # Use 0 since we have batch size 1
    
    # Extract the gradients of the top predicted class w.r.t. conv layer
    grads = tape.gradient(loss, conv_outputs)
    
    if grads is None:
        raise ValueError("Could not compute gradients")
    
    # Pool the gradients across the spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the conv layer output with the computed gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize the heatmap between 0 & 1 for visualization
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()
    
    # Create a model that maps the input image to the activations of the last conv layer
    # as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(layer_name).output, model.output]
    )
    
    # Compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, class_index]
    
    # Extract the gradients of the top predicted class w.r.t. conv layer
    grads = tape.gradient(loss, conv_outputs)
    
    # Pool the gradients across the spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the conv layer output with the computed gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize the heatmap between 0 & 1 for visualization
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def overlay_gradcam(image, heatmap, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on the original image
    
    Args:
        image: Original image (numpy array)
        heatmap: Grad-CAM heatmap
        alpha: Transparency factor for overlay
    
    Returns:
        overlayed_image: Image with Grad-CAM overlay
    """
    
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert heatmap to RGB
    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
    
    # Ensure image is in float format and normalized
    if image.dtype == tf.uint8 or image.max() > 1:
        image_normalized = image.astype(np.float32) / 255.0
    else:
        image_normalized = image.astype(np.float32)
    
    # Overlay heatmap on image
    overlayed = heatmap_colored * alpha + image_normalized * (1 - alpha)
    
    return overlayed

def main():
    # Load the model
    model_path = 'weight/weight.keras'
    print(f"Loading model from: {model_path}")
    
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        
        # Debug: Print model layers to understand structure
        print("\n🔍 Model layers:")
        for i, layer in enumerate(model.layers):
            layer_type = layer.__class__.__name__
            if hasattr(layer, 'output_shape'):
                output_shape = layer.output_shape
            else:
                output_shape = "No output_shape"
            print(f"  {i}: {layer.name} ({layer_type}) - {output_shape}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Get test images
    test_images_dir = 'test_images'
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    
    test_images = []
    for file in os.listdir(test_images_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            test_images.append(os.path.join(test_images_dir, file))
    
    if not test_images:
        print("❌ No test images found!")
        return
    
    print(f"Found {len(test_images)} test images:")
    for img in test_images:
        print(f"  - {img}")
    
    # Make predictions
    results = []
    print("\n🔍 Making predictions...")
    
    for image_path in test_images:
        try:
            predicted_class, confidence, probabilities, original_image, processed_image, heatmap, gradcam_overlay = predict_image(model, image_path)
            results.append((image_path, predicted_class, confidence, probabilities, original_image, processed_image, heatmap, gradcam_overlay))
            
            print(f"\n📸 {os.path.basename(image_path)}:")
            print(f"   Prediction: {predicted_class}")
            print(f"   Confidence: {confidence:.2%}")
            print("   All probabilities:")
            for i, class_name in enumerate(class_names):
                print(f"     {class_name}: {probabilities[i]:.2%}")
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
    
    # Visualize results
    if results:
        print("\n📊 Generating visualization...")
        visualize_predictions(results)
        print("✅ Visualization saved as 'predictions.png'")

if __name__ == "__main__":
    main()