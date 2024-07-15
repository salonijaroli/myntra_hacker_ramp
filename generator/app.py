from flask import Flask, request, jsonify
from diffusers import StableDiffusionPipeline
import torch
import base64
from io import BytesIO

app = Flask(__name__)

# Load the model safely
model = None
try:
    model = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    if torch.cuda.is_available():
        model = model.to("cuda")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")


@app.route('/generateimages/<prompt>', methods=['GET'])
def generate_images(prompt):
    try:
        # Generate the image from the prompt
        images = model(prompt).images

        # Convert images to base64 strings
        image_data = []
        for image in images:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_data.append({"url": f"data:image/png;base64,{img_str}"})

        return jsonify({"data": image_data})
    except Exception as e:
        print(f"Error generating image: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/')
def index():
    return "Hello, World!"


if __name__ == '__main__':
    app.run(debug=True)
