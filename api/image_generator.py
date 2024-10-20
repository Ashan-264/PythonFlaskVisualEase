from flask import Flask, request, jsonify
from flask_cors import CORS
from gradio_client import Client
import shutil
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()  # This ensures the .env file is loaded

# Verify that the tokens are loaded (for debugging purposes)
print(f"GROQ_API_KEY: {os.getenv('GROQ_API_KEY')}")
print(f"HF_TOKEN: {os.getenv('HF_TOKEN')}")
print(f"BLOB_READ_WRITE_TOKEN: {os.getenv('BLOB_READ_WRITE_TOKEN')}")

app = Flask(__name__)
CORS(app)

# Setup client for the Huggingface model
hf_token = os.getenv("HF_TOKEN")
client = Client("black-forest-labs/FLUX.1-schnell", hf_token=hf_token)

save_directory = "/tmp/generated_images"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

BLOB_RW_TOKEN = os.getenv("BLOB_READ_WRITE_TOKEN")

@app.route('/api/image_generator', methods=['POST'])
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

        webp_image_path, _ = result

        # Upload the WebP image directly to Vercel Blob
        with open(webp_image_path, "rb") as img_file:
            response = requests.put(
                "https://blob.vercel-storage.com/upload?filename=generated_image.webp",
                headers={
                    "Authorization": f"Bearer {BLOB_RW_TOKEN}",
                    "Content-Type": "image/webp"
                },
                data=img_file
            )

        if response.status_code != 200:
            return jsonify({
                'error': 'Failed to upload image to Vercel Blob',
                'details': response.text
            }), 500

        original_url = response.json().get("url")
        friendly_url = original_url.replace("upload-", "generated_image-")

        return jsonify({'imageUrl': friendly_url})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
