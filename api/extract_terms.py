from flask import Flask, request, jsonify
from dotenv import load_dotenv
from groq import Groq
import os
from flask_cors import CORS
import logging

# Load environment variables
load_dotenv()

# Initialize Flask app and CORS
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# Retrieve the API key and set up the Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

@app.route('/api/extract_terms', methods=['POST'])
def extract_terms():
    try:
        # Get JSON data from the request
        data = request.json
        logging.info(f"Received request data: {data}")

        # Extract text and study level from the request
        text = data.get('text')
        study_level = data.get('level')

        # Validate that both text and study level are provided
        if not text or not study_level:
            return jsonify({"error": "Both 'text' and 'level' are required fields."}), 400

        # Create the prompt for term extraction based on the study level and text
        prompt = f"Extract uncommon terms and definitions from the following text for a {study_level} student: {text}"
        logging.info(f"Generated prompt: {prompt}")

        # Call the Groq API to generate the term extraction response
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192"
        )

        # Extract the result from the API response
        result = response.choices[0].message.content
        logging.info(f"Generated result: {result}")

        # Return the extracted terms and definitions as JSON
        return jsonify({"terms_and_definitions": result}), 200

    except Exception as e:
        # Log the error and return a 500 error with the message
        logging.error(f"Error while extracting terms: {e}")
        return jsonify({"error": str(e)}), 500

# Run the Flask app if this script is executed directly
if __name__ == '__main__':
    app.run(debug=True)
