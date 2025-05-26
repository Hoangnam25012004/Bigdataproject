from flask import Flask, request, jsonify, render_template, make_response
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import re
import string
import emoji
import os
from waitress import serve
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["null", "file://", "http://localhost:5000", "http://127.0.0.1:5000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept", "Origin"]
    }
})

# Check if model and tokenizer files exist
MODEL_PATH = 'sentiment_lstm_model.h5'
TOKENIZER_PATH = 'tokenizer.pickle'
MAX_LEN = 30

try:
    print("Loading model and tokenizer...")
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
    print("Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"Error loading model or tokenizer: {str(e)}")
    print("Please make sure you have run bigdataproj.py first to generate the model files.")
    model = None
    tokenizer = None

def pre_process(text):
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = emoji.demojize(text)
    return text.strip()

# Sentiment mapping
label_mapping = {0: 'Positive', 1: 'Neutral', 2: 'Negative', 3: 'Irrelevant'}

@app.after_request
def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = request.headers.get('Origin', '*')
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept, Origin'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    return response

@app.route('/')
def home():
    return render_template('sentiment_analysis.html')

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
        
    if model is None or tokenizer is None:
        return jsonify({
            'error': 'Model not loaded. Please run bigdataproj.py first to generate the model.'
        }), 503

    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400

        # Preprocess the text
        clean_text = pre_process(text)
        
        # Convert to sequence and pad
        seq = tokenizer.texts_to_sequences([clean_text])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
        
        # Get prediction
        pred = model.predict(padded)
        label = label_mapping[np.argmax(pred)]
        confidence = float(np.max(pred))
        
        return jsonify({
            'sentiment': label,
            'confidence': confidence,
            'processed_text': clean_text
        })

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("\nStarting server...")
    print("Please access the application by opening sentiment_analysis.html in Chrome")
    print("Server API endpoint: http://127.0.0.1:5000")
    
    # Use Waitress WSGI server
    serve(app, host='127.0.0.1', port=5000) 