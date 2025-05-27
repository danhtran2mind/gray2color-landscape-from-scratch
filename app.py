import gradio as gr
from PIL import Image
import os
import numpy as np
import tensorflow as tf
import requests
from skimage.color import lab2rgb

from models.auto_encoder_gray2color import SpatialAttention

WIDTH, HEIGHT = 512, 512

# Load the saved model once at startup
load_model_path = "./ckpts/best_model.h5"
if not os.path.exists(load_model_path):
    os.makedirs(os.path.dirname(load_model_path), exist_ok=True)
    url = "https://huggingface.co/danhtran2mind/autoencoder-grayscale2color-landscape/resolve/main/ckpts/best_model.h5"
    print(f"Downloading model from {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(load_model_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Download complete.")

print(f"Loading model from {load_model_path}...")
loaded_autoencoder = tf.keras.models.load_model(
    load_model_path,
    custom_objects={'SpatialAttention': SpatialAttention}
)

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

custom_css = """
body {background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%) !important;}
.gradio-container {background: transparent !important;}
h1, .gr-title {color: #007bff !important; font-family: 'Segoe UI', sans-serif;}
.gr-description {color: #333333 !important; font-size: 1.1em;}
.gr-input, .gr-output {border-radius: 18px !important; box-shadow: 0 4px 24px rgba(0,0,0,0.1);}
.gr-button {background: linear-gradient(90deg, #007bff 0%, #00c4cc 100%) !important; color: #fff !important; border: none !important; border-radius: 12px !important;}
"""

demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", label="Upload Grayscale Landscape", image_mode="L"),
    outputs=gr.Image(type="pil", label="Colorized Output"),
    title="🌄 Gray2Color Landscape Autoencoder",
    description=(
        "<div style='font-size:1.15em;line-height:1.6em;'>"
        "Transform your <b>grayscale landscape</b> photos into vivid color with a state-of-the-art autoencoder.<br>"
        "Simply upload a grayscale image and see the magic happen!"
        "</div>"
    ),
    theme="soft",
    css=custom_css,
    allow_flagging="never",
    examples=[
        ["examples/example_input_1.jpg", "examples/example_output_1.jpg"],
        ["examples/example_input_2.jpg", "examples/example_output_2.jpg"]
    ]
)

if __name__ == "__main__":
    demo.launch()
