# File: api/extract_terms.py
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from groq import Groq
import os
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

@app.route('/api/extract_terms', methods=['POST'])
def extract_terms():
    data = request.json
    text = data.get('text')
    study_level = data.get('level')

    prompt = f"Extract uncommon terms and definitions from the following text for a {study_level} student: {text}"

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-70b-8192"
    )

    result = response.choices[0].message.content
    #return jsonify({"terms_and_definitions": result})
    return "hi"
