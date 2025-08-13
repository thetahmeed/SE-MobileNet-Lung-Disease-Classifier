#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64

# Set random seeds for reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Class names from the training notebook
class_names = ['covid', 'normal', 'pneumonia']

# Load the model once when the app starts
MODEL_PATH = 'weight/weight.keras'
model = None

def load_model():
    """Load the trained model"""
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

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

def predict_image(image):
    """Make prediction on an uploaded image"""
    
    if model is None:
        return None, "Model not loaded. Please restart the application."
    
    try:
        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        
        # Ensure 3 channels (RGB)
        if len(image_array.shape) == 2:
            image_array = np.stack([image_array] * 3, axis=-1)
        elif image_array.shape[-1] == 4:  # RGBA
            image_array = image_array[:, :, :3]
        
        # Convert to tensor
        image_tensor = tf.convert_to_tensor(image_array)
        
        # Preprocess the image
        processed_image = preprocess(image_tensor)
        
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
        
        # Create probability dictionary for gradio
        prob_dict = {}
        for i, class_name in enumerate(class_names):
            prob_dict[class_name] = float(probabilities[i].numpy())
        
        # Create result text
        result_text = f"""
## COVID-19 Chest X-Ray Classification Results

### **Prediction: {predicted_class.upper()}**
### **Confidence: {confidence:.2%}**

### **Detailed Probabilities:**
- **COVID-19**: {probabilities[0].numpy():.2%}
- **Normal**: {probabilities[1].numpy():.2%}
- **Pneumonia**: {probabilities[2].numpy():.2%}

### **Interpretation:**
"""
        
        if predicted_class == 'covid':
            result_text += "**COVID-19 detected** - Please consult with a healthcare professional for proper diagnosis and treatment."
        elif predicted_class == 'normal':
            result_text += "**Normal chest X-ray** - No significant abnormalities detected."
        elif predicted_class == 'pneumonia':
            result_text += "**Pneumonia detected** - Please consult with a healthcare professional for proper diagnosis and treatment."
        
        result_text += ""
        
        return prob_dict, result_text
        
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

def create_interface():
    """Create the Gradio interface"""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        max-width: 1200px;
        margin: auto;
    }
    .main-header {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        color: #856404;
    }
    """
    
    # Create the interface
    with gr.Blocks(css=custom_css, title="Chest X-Ray Classifier") as demo:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>Chest X-Ray Classification</h1>
            <p>Upload a chest X-ray image to classify it as COVID-19, Normal, or Pneumonia</p>
        </div>
        """)
        
      
        
        with gr.Row():
            # Left column - Input
            with gr.Column(scale=1):
                gr.HTML("<h3>Upload Image</h3>")
                
                image_input = gr.Image(
                    label="Upload Chest X-Ray Image",
                    type="pil",
                    height=400
                )
                
                predict_btn = gr.Button(
                    "Analyze Image", 
                    variant="primary",
                    size="lg"
                )
                
                # Model info
                # gr.HTML("""
                # <div style="margin-top: 20px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
                #     <h4>📋 Model Information</h4>
                #     <ul>
                #         <li><strong>Architecture:</strong> MobileNetV2</li>
                #         <li><strong>Input Size:</strong> 224x224 pixels</li>
                #         <li><strong>Classes:</strong> COVID-19, Normal, Pneumonia</li>
                #         <li><strong>Training:</strong> Custom dataset with data augmentation</li>
                #     </ul>
                # </div>
                # """)
            
            # Right column - Results
            with gr.Column(scale=1):
                gr.HTML("<h3>Results</h3>")
                
                # Probability plot
                probability_plot = gr.Label(
                    label="Classification Probabilities",
                    num_top_classes=3
                )
                
                # Detailed results
                result_text = gr.Markdown(
                    value="Upload an image and click 'Analyze Image' to see results.",
                    label="Detailed Analysis"
                )
        
        # Set up the event handler
        predict_btn.click(
            fn=predict_image,
            inputs=image_input,
            outputs=[probability_plot, result_text]
        )
    
    return demo

def main():
    """Main function to run the Gradio app"""
    
    # Load the model
    if not load_model():
        print("Failed to load model. Please check the model path and try again.")
        return
    
    # Create and launch the interface
    demo = create_interface()
    
    print("Launching Gradio interface...")
    
    # Launch the app
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True to create a public link
        debug=False,
        show_error=True
    )

if __name__ == "__main__":
    main() 