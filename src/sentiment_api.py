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
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["null", "file://", "http://localhost:5000", "http://127.0.0.1:5000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept", "Origin"]
    }
})

# Check if model, tokenizer, and sentiment words files exist
MODEL_PATH = 'sentiment_lstm_model.h5'
TOKENIZER_PATH = 'tokenizer.pickle'
SENTIMENT_WORDS_PATH = 'sentiment_words.pickle'
MAX_LEN = 50

try:
    print("Loading model, tokenizer, and sentiment words...")
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open(SENTIMENT_WORDS_PATH, 'rb') as handle:
        sentiment_words = pickle.load(handle)
    print("Model, tokenizer, and sentiment words loaded successfully!")
except Exception as e:
    print(f"Error loading model, tokenizer, or sentiment words: {str(e)}")
    print("Please make sure you have run train_model.py first to generate all necessary files.")
    model = None
    tokenizer = None
    sentiment_words = None

def pre_process(text):
    """
    Preprocess text using discovered sentiment words
    """
    if text is None:
        return ""
    
    # Convert emojis to text
    text = emoji.demojize(text)
    text = text.lower()
    
    # Replace contractions
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'m", " am", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'d", " would", text)
    
    # Remove URLs and clean up
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Preserve important punctuation
    text = re.sub(r'[!]+', ' ! ', text)
    text = re.sub(r'[?]+', ' ? ', text)
    
    # Tokenize
    words = word_tokenize(text)
    
    # Process multi-word expressions and sentiment indicators
    processed_words = []
    i = 0
    while i < len(words):
        word = words[i]
        next_word = words[i + 1] if i + 1 < len(words) else None
        
        # Check for negation patterns
        if word in ['not', 'never'] and next_word:
            if next_word in sentiment_words['positive']:
                processed_words.append('not_positive')
            elif next_word in sentiment_words['negative']:
                processed_words.append('not_negative')
            else:
                processed_words.extend([word, next_word])
            i += 2
            continue
        
        # Check for intensifier patterns
        if word in sentiment_words['intensifiers'] and next_word:
            if next_word in sentiment_words['positive']:
                processed_words.append('very_positive')
            elif next_word in sentiment_words['negative']:
                processed_words.append('very_negative')
            else:
                processed_words.extend([word, next_word])
            i += 2
            continue
        
        processed_words.append(word)
        i += 1
    
    text = ' '.join(processed_words)
    
    # Clean up final text
    text = re.sub(r'[^\w\s!?]', '', text)
    text = re.sub(r'\s+', ' ', text)
    
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
        
    if model is None or tokenizer is None or sentiment_words is None:
        return jsonify({
            'error': 'Model not loaded. Please run train_model.py first to generate all necessary files.'
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
        
        # Add debug information
        all_sentiments = {label_mapping[i]: float(pred[0][i]) for i in range(len(pred[0]))}
        
        # Add preprocessing debug info
        debug_info = {
            'original_text': text,
            'processed_text': clean_text,
            'tokens': clean_text.split(),
            'sequence_length': len(seq[0]),
            'detected_negative_words': [word for word in clean_text.split() if word in sentiment_words['negative']],
            'detected_positive_words': [word for word in clean_text.split() if word in sentiment_words['positive']],
            'detected_neutral_words': [word for word in clean_text.split() if word in sentiment_words['neutral']],
            'detected_intensifiers': [word for word in clean_text.split() if word in sentiment_words['intensifiers']]
        }
        
        return jsonify({
            'sentiment': label,
            'confidence': confidence,
            'all_sentiments': all_sentiments,
            'debug_info': debug_info
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