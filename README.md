# Autoencoder Ggrayscale2Color Landscape From Scratch

## Dataset
See at [README Dataset](./dataset/README.md)

## Demostration
{HugginFace Demo](https://huggingface.co/spaces/danhtran2mind/autoencoder-grayscale2color-landscape)

## Metrics
PSNR
- Validation set: 21.70

## Usage

### Download Model
```bash
git clone https://huggingface.co/danhtran2mind/autoencoder-grayscale2color-landscape
```
```bash
cd autoencoder-grayscale2color-landscape
git lfs pull
```
### Import Libraries
```python
from PIL import Image
import os
import numpy as np
import tensorflow as tf
import requests
from skimage.color import lab2rgb
import matplotlib.pyplot as plt
from models.auto_encoder_gray2color import SpatialAttention
```
### Load Model file
```python
# Load the saved model once at startup
load_model_path = "./ckpts/best_model.h5"

print(f"Loading model from {load_model_path}...")
loaded_autoencoder = tf.keras.models.load_model(
    load_model_path,
    custom_objects={'SpatialAttention': SpatialAttention}
)
```

### Define Functions
```python
def process_image(input_img):
    # Store original input dimensions
    original_width, original_height = input_img.size

    # Convert PIL Image to grayscale and resize to model input size
    img = input_img.convert("L")  # Convert to grayscale (single channel)
    img = img.resize((WIDTH, HEIGHT))  # Resize to 512x512 for model
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize to [0, 1]
    img_array = img_array[None, ..., 0:1]  # Add batch dimension, shape: (1, 512, 512, 1)

    # Run inference (assuming loaded_autoencoder predicts a*b* channels)
    output_array = loaded_autoencoder.predict(img_array)  # Shape: (1, 512, 512, 2) for a*b*
    print("output_array shape: ", output_array.shape)

    # Extract L* (grayscale input) and a*b* (model output)
    L_channel = img_array[0, :, :, 0] * 100.0  # Denormalize L* to [0, 100]
    ab_channels = output_array[0] * 128.0  # Denormalize a*b* to [-128, 128]

    # Combine L*, a*, b* into a 3-channel L*a*b* image
    lab_image = np.stack([L_channel, ab_channels[:, :, 0], ab_channels[:, :, 1]], axis=-1)  # Shape: (512, 512, 3)

    # Convert L*a*b* to RGB
    rgb_array = lab2rgb(lab_image)  # Convert to RGB, output in [0, 1]
    rgb_array = np.clip(rgb_array, 0, 1) * 255.0  # Scale to [0, 255]
    rgb_image = Image.fromarray(rgb_array.astype(np.uint8), mode="RGB")  # Create RGB PIL image

    # Resize output image to match input image resolution
    rgb_image = rgb_image.resize((original_width, original_height), Image.Resampling.LANCZOS)

    return rgb_image

def process_and_plot_images(input_path):
    # Read input image
    input_img = Image.open(input_path)
    
    # Process the image (placeholder for your process_image function)
    output_img = process_image(input_img)
    
    # Save output image to output.jpg
    output_img.save("output.jpg")
    
    return input_img, output_img

def plot_in_out_images(input_img, output_img):
    # Create a figure with two subplots for input and output images
    plt.figure(figsize=(17, 8), dpi=300)  # Set DPI to 300
    
    # Plot input image
    plt.subplot(1, 2, 1)
    plt.imshow(input_img, cmap='gray')
    plt.title("Input Image")
    plt.axis('off')  # Hide axes for cleaner display
    
    # Plot output image
    plt.subplot(1, 2, 2)
    plt.imshow(output_img, cmap='gray')
    plt.title("Output Image")
    plt.axis('off')  # Hide axes for cleaner display
    
    # Save the figure as output.jpg with 300 DPI
    plt.savefig("output.jpg", dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()
```
### Inference
```python
# Example usage
WIDTH, HEIGHT = 512, 512
# Replace 'input_image.jpg' with the path to your image
image_path = "<input_image.jpg>"
input_img, output_img = process_and_plot_images(image_path)

plot_in_out_images(input_img, output_img)
```
### Example Output
![Plot Image](./examples/model_output.jpg)

