import os
import requests
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from gradio_client import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Setup Huggingface client
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    logging.error("HF_TOKEN is not set in environment")
client = Client("black-forest-labs/FLUX.1-schnell", hf_token=hf_token)

BLOB_READ_WRITE_TOKEN = os.getenv("BLOB_READ_WRITE_TOKEN")
if not BLOB_READ_WRITE_TOKEN:
    logging.error("BLOB_READ_WRITE_TOKEN is not set in environment")

# Ensure temporary directory exists for generated images
save_directory = "/tmp/generated_images"
os.makedirs(save_directory, exist_ok=True)
logging.debug(f"Ensured save_directory exists: {save_directory}")

@app.route('/api/image_generator', methods=['POST'])
def generate_image():
    logging.info("Received /api/image_generator request")
    data = request.json
    logging.debug(f"Request JSON payload: {data}")

    prompt = data.get('textPart')
    if not prompt:
        logging.warning("No prompt provided in request")
        return jsonify({'error': 'Prompt is required'}), 400

    logging.info(f"Generating image for prompt: {prompt!r}")
    try:
        # 1) Call HF via gradio-client
        result = client.predict(
            prompt=prompt,
            seed=0,
            randomize_seed=True,
            width=1024,
            height=1024,
            num_inference_steps=4,
            api_name="/infer"
        )
        logging.debug(f"Huggingface client.predict returned: {result}")

        webp_image_path, _ = result
        if not os.path.exists(webp_image_path):
            logging.error(f"Generated file not found at: {webp_image_path}")
            return jsonify({'error': 'Image file missing after generation'}), 500
        logging.info(f"Image generated on disk: {webp_image_path}")

        # 2) Upload the WebP image
        with open(webp_image_path, "rb") as img_file:
            logging.info("Uploading image to Vercel Blob Storage")
            response = requests.put(
                "https://blob.vercel-storage.com/upload?filename=generated_image.webp",
                headers={
                    "Authorization": f"Bearer {BLOB_READ_WRITE_TOKEN}",
                    "Content-Type": "image/webp"
                },
                data=img_file
            )

        logging.debug(f"Blob Upload HTTP {response.status_code}: {response.text}")
        if response.status_code != 200:
            logging.error("Failed to upload image to Vercel Blob")
            return jsonify({
                'error': 'Failed to upload image to Vercel Blob',
                'details': response.text
            }), 500

        blob_json = response.json()
        image_url = blob_json.get("url")
        if not image_url:
            logging.error("No URL in Blob response JSON")
            return jsonify({'error': 'Image URL not found in Blob response.'}), 500

        logging.info(f"Successfully uploaded image, URL: {image_url}")
        return jsonify({'imageUrl': image_url})

    except Exception as e:
        logging.exception("Unhandled exception in generate_image")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 3000))
    logging.info(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port)
