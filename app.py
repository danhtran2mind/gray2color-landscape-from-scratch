import gradio as gr
from PIL import Image
import os
import tensorflow as tf
import requests

from .models.auto_encoder_gray2color import SpatialAttention

# Load the saved model once at startup
load_model_path = "./ckpts/best_model.h5"
if not os.path.exists(load_model_path):
    os.makedirs(os.path.dirname(load_model_path), exist_ok=True)
    url = "https://huggingface.co/danhtran2mind/autoencoder-grayscale-to-color-landscape/resolve/main/ckpts/best_model.h5"
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
    # Convert PIL Image to numpy array and normalize
    img = input_img.convert("RGB")
    img = img.resize((256, 256))  # adjust size as needed
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = img_array[None, ...]  # add batch dimension

    # Run inference
    output_array = loaded_autoencoder.predict(img_array)
    output_img = tf.keras.preprocessing.image.array_to_img(output_array[0])

    return output_img

custom_css = """
body {background: linear-gradient(135deg, #232526 0%, #414345 100%) !important;}
.gradio-container {background: transparent !important;}
h1, .gr-title {color: #00e6d3 !important; font-family: 'Segoe UI', sans-serif;}
.gr-description {color: #e0e0e0 !important; font-size: 1.1em;}
.gr-input, .gr-output {border-radius: 18px !important; box-shadow: 0 4px 24px rgba(0,0,0,0.18);}
.gr-button {background: linear-gradient(90deg, #00e6d3 0%, #0072ff 100%) !important; color: #fff !important; border: none !important; border-radius: 12px !important;}
"""

demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", label="Upload Grayscale Landscape", image_mode="L", shape=(256, 256)),
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
        ["examples/grayscale_landscape1.jpg"],
        ["examples/grayscale_landscape2.jpg"]
    ]
)

if __name__ == "__main__":
    demo.launch()