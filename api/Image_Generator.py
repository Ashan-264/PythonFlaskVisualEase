# File: api/image_generator.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from gradio_client import Client
import shutil
import os

app = Flask(__name__)
CORS(app)

hf_token = os.getenv("HF_TOKEN")
client = Client("black-forest-labs/FLUX.1-schnell", hf_token=hf_token)

save_directory = "generated_images"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

@app.route('/api/generate-image', methods=['POST'])
def generate_image():
    data = request.json
    prompt = data.get('textPart')
    
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    try:
        result = client.predict(
            prompt=prompt,
            seed=0,
            randomize_seed=True,
            width=1024,
            height=1024,
            num_inference_steps=4,
            api_name="/infer"
        )

        image_path, _ = result
        save_path = os.path.join(save_directory, "generated_image.webp")
        shutil.copy(image_path, save_path)

        return jsonify({'imageUrl': f'/api/generated_images/generated_image.webp'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generated_images/<filename>')
def send_image(filename):
    return send_from_directory(save_directory, filename)

