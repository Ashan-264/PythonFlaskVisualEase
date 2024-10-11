from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from gradio_client import Client
import shutil
import os

app = Flask(__name__)
CORS(app)

hf_token = os.getenv("HF_TOKEN")
client = Client("black-forest-labs/FLUX.1-schnell", hf_token=hf_token)

# Use the writable temporary directory provided by Vercel.
save_directory = "/tmp/generated_images"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

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

        # Assume result is a tuple where the first element is the image path.
        image_path, _ = result
        save_path = os.path.join(save_directory, "generated_image.webp")
        shutil.copy(image_path, save_path)

        # Send the generated image as a downloadable file.
        return send_file(save_path, as_attachment=True, download_name="generated_image.webp")

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
