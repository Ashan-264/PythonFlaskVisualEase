import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS from Flask-CORS
from gradio_client import Client
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all origins
CORS(app)  # This allows any origin to make requests

# Setup Huggingface client
hf_token = os.getenv("HF_TOKEN")
client = Client("black-forest-labs/FLUX.1-schnell", hf_token=hf_token)

BLOB_RW_TOKEN = os.getenv("BLOB_RW_TOKEN")

# Ensure temporary directory exists for generated images
save_directory = "/tmp/generated_images"
os.makedirs(save_directory, exist_ok=True)

@app.route('/api/image_generator', methods=['POST'])
def generate_image():
    data = request.json
    prompt = data.get('textPart')

    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    try:
        # Generate the image with Huggingface API
        result = client.predict(
            prompt=prompt,
            seed=0,
            randomize_seed=True,
            width=1024,
            height=1024,
            num_inference_steps=4,
            api_name="/infer"
        )

        webp_image_path, _ = result

        # Upload the WebP image to Vercel Blob Storage
        with open(webp_image_path, "rb") as img_file:
            response = requests.put(
                "https://blob.vercel-storage.com/upload?filename=generated_image.webp",
                headers={
                    "Authorization": f"Bearer {BLOB_RW_TOKEN}",
                    "Content-Type": "image/webp"
                },
                data=img_file
            )

        # Log the response for debugging
        print(f"Blob Upload Response: {response.json()}")

        if response.status_code != 200:
            return jsonify({
                'error': 'Failed to upload image to Vercel Blob',
                'details': response.text
            }), 500

        # Extract the image URL from the Blob response
        image_url = response.json().get("url")

        if not image_url:
            raise ValueError("Image URL not found in Blob response.")

        # Return the image URL to the frontend
        return jsonify({'imageUrl': image_url})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
