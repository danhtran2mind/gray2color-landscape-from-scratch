import gradio as gr
from PIL import Image
import os
import numpy as np
import tensorflow as tf
import requests
from skimage.color import lab2rgb
from models.autoencoder_gray2color import SpatialAttention
from models.unet_gray2color import SelfAttentionLayer

# Set float32 policy
tf.keras.mixed_precision.set_global_policy('float32')

# Model-specific input shapes
MODEL_INPUT_SHAPES = {
    "autoencoder": (512, 512),
    "unet": (1024, 1024),
    "transformer": (1024, 1024)
}

# Define model paths
load_model_paths = [
    "./ckpts/autoencoder/autoencoder_colorization_model.h5",
    "./ckpts/unet/unet_colorization_model.keras",
    "./ckpts/transformer/transformer_colorization_model.keras"
]

# Load models at startup
models = {}
print("Loading models...")
for path in load_model_paths:
    model_name = os.path.basename(os.path.dirname(path))
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url_map = {
            "autoencoder": "ckpts/best_model.h5",
            "unet": "ckpts/unet_colorization_model.keras",  # Replace with valid URL
            "transformer": "ckpts/transformer_colorization_model.keras"  # Replace with valid URL
        }
        if model_name in url_map:
            print(f"Downloading {model_name} model from {url_map[model_name]}...")
            with requests.get(url_map[model_name], stream=True) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Download complete for {model_name}.")
    
    custom_objects = {
        "autoencoder": {'SpatialAttention': SpatialAttention},
        "unet": {'SelfAttentionLayer': SelfAttentionLayer},
        "transformer": None
    }
    print(f"Loading {model_name} model from {path}...")
    models[model_name] = tf.keras.models.load_model(
        path,
        custom_objects=custom_objects[model_name],
        compile=False
    )
    models[model_name].compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=7e-5),
        loss=tf.keras.losses.MeanSquaredError()
    )
    print(f"{model_name} model loaded.")

print("All models loaded.")

def process_image(input_img, model_name):
    # Store original input dimensions
    original_width, original_height = input_img.size
    # Get model-specific input shape
    width, height = MODEL_INPUT_SHAPES[model_name.lower()]
    # Convert PIL Image to grayscale and resize to model input size
    img = input_img.convert("L")
    img = img.resize((width, height))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = img_array[None, ..., 0:1]  # Shape: (1, height, width, 1)
    
    # Select model
    selected_model = models[model_name.lower()]
    # Run inference
    output_array = selected_model.predict(img_array)  # Shape: (1, height, width, 2)
    
    # Extract L* and a*b*
    L_channel = img_array[0, :, :, 0] * 100.0  # Denormalize L* to [0, 100]
    ab_channels = output_array[0] * 128.0  # Denormalize a*b* to [-128, 128]
    
    # Combine L*, a*, b*
    lab_image = np.stack([L_channel, ab_channels[:, :, 0], ab_channels[:, :, 1]], axis=-1)
    
    # Convert to RGB
    rgb_array = lab2rgb(lab_image)
    rgb_array = np.clip(rgb_array, 0, 1) * 255.0
    rgb_image = Image.fromarray(rgb_array.astype(np.uint8), mode="RGB")
    
    # Resize output to original resolution
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
    inputs=[
        gr.Image(type="pil", label="Upload Grayscale Landscape", image_mode="L"),
        gr.Dropdown(
            choices=["Autoencoder", "Unet", "Transformer"],
            label="Select Model",
            value="Autoencoder"
        )
    ],
    outputs=gr.Image(type="pil", label="Colorized Output"),
    title="Grayscale2Color Landscape from scratch🌄",
    description=(
        "<div style='font-size:1.15em;line-height:1.6em;'>"
        "Transform your <b>grayscale landscape</b> photos into vivid color using advanced deep learning models.<br>"
        "Upload a grayscale image, select a model (Autoencoder, U-Net, or Transformer), and see the results!"
        "</div>"
    ),
    theme="soft",
    css=custom_css,
    allow_flagging="never",
    examples=[
        ["assets/input/input_1.jpg", "assets/autoencoder/autoencoder_output_1.jpg", "Autoencoder"],
        ["assets/input/input_2.jpg", "assets/autoencoder/autoencoder_output_2.jpg", "Autoencoder"],
        ["assets/input/input_1.jpg", "assets/unet/unet_output_1.jpg", "Unet"],
        ["assets/input/input_2.jpg", "assets/unet/unet_output_2.jpg", "Unet"],
        ["assets/input/input_1.jpg", "assets/transformer/transformer_output_1.jpg", "Transformer"],
        ["assets/input/input_2.jpg", "assets/transformer/transformer_output_2.jpg", "Transformer"]
    
    ]
)

if __name__ == "__main__":
    demo.launch()