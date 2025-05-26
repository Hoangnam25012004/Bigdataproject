from flask import Flask, request, jsonify, render_template, make_response
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType
from pyspark.ml import PipelineModel
import re
import string
import emoji
import os
import sys
import pickle
from waitress import serve
from flask_cors import CORS
import nltk
import traceback
nltk.download('punkt')

# Set up Spark environment
os.environ['HADOOP_HOME'] = "C:\\hadoop"
os.environ['SPARK_LOCAL_DIRS'] = 'C:\\temp'
os.environ['JAVA_HOME'] = "C:\\Program Files\\Java\\jdk-17"

if os.name == 'nt':  # Windows
    os.environ['PATH'] = os.environ['PATH'] + ';' + os.environ['HADOOP_HOME'] + '\\bin'
    
    # Get Python executable path
    python_path = sys.executable
    python_dir = os.path.dirname(python_path)
    
    # Add Python to system PATH
    os.environ['PATH'] = f"{python_dir};{python_dir}\\Scripts;{os.environ['PATH']}"
    os.environ['PYSPARK_PYTHON'] = python_path
    os.environ['PYSPARK_DRIVER_PYTHON'] = python_path

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept", "Origin"]
    }
})

# Initialize Spark session
spark = SparkSession.builder \
    .appName("SentimentAnalysisAPI") \
    .config("spark.driver.host", "127.0.0.1") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .config("spark.python.worker.reuse", "true") \
    .config("spark.python.executable", sys.executable) \
    .master("local[1]") \
    .getOrCreate()

# Suppress INFO and WARN messages
spark.sparkContext.setLogLevel("ERROR")

# Check if model and vocabulary files exist
MODEL_PATH = 'spark_sentiment_model'
VOCAB_PATH = 'spark_sentiment_vocab.pickle'

try:
    print("Loading model and vocabulary...")
    model = PipelineModel.load(MODEL_PATH)
    with open(VOCAB_PATH, 'rb') as handle:
        vocab_data = pickle.load(handle)
    print("Model and vocabulary loaded successfully!")
except Exception as e:
    print(f"Error loading model or vocabulary: {str(e)}")
    print("Please make sure you have run train_model.py first to generate all necessary files.")
    model = None
    vocab_data = None

def pre_process(text):
    """Text preprocessing function"""
    if text is None:
        return ""
    
    # Convert emojis to text
    text = emoji.demojize(text)
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# Register UDF for preprocessing
pre_process_udf = udf(pre_process, StringType())

@app.after_request
def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
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
        
    if model is None or vocab_data is None:
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

        # Create a single-row DataFrame with the input text
        input_df = spark.createDataFrame([(text,)], ["tweet"])
        
        # Preprocess and make prediction
        input_df = input_df.withColumn("clean_tweet", pre_process_udf(col("tweet")))
        
        try:
            prediction = model.transform(input_df)
            
            # Get prediction and probability scores
            result = prediction.select("prediction", "probability").collect()[0]
            pred_label = int(result["prediction"])  # Convert to int
            probabilities = result["probability"].toArray()
            
            # Ensure the prediction index exists in label mapping
            if pred_label not in vocab_data['label_mapping']:
                print(f"Warning: Unexpected prediction index {pred_label}")
                # Find the highest probability and use its index
                pred_label = int(probabilities.argmax())
            
            # Get the text label
            label = vocab_data['label_mapping'].get(pred_label, "Unknown")
            confidence = float(probabilities[pred_label])
            
            # Create all sentiments dictionary with safety checks
            all_sentiments = {}
            for i, prob in enumerate(probabilities):
                if i in vocab_data['label_mapping']:
                    all_sentiments[vocab_data['label_mapping'][i]] = float(prob)
                else:
                    print(f"Warning: Missing label mapping for index {i}")
            
            # Get words from vocabulary that appear in the text
            processed_text = pre_process(text)
            words = set(processed_text.split())
            vocab_words = set(vocab_data['vocabulary'])
            found_words = words.intersection(vocab_words)
            
            # Add debug information
            debug_info = {
                'original_text': text,
                'processed_text': processed_text
            }
            
            return jsonify({
                'sentiment': label,
                'confidence': confidence,
                'all_sentiments': all_sentiments,
                'debug_info': debug_info
            })
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            print("Traceback:")
            traceback.print_exc()
            return jsonify({'error': f'Prediction error: {str(e)}'}), 500

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    print("\nStarting server...")
    print("Please access the application by opening sentiment_analysis.html in Chrome")
    print("Server API endpoint: http://127.0.0.1:5000")
    
    # Use Waitress WSGI server
    serve(app, host='127.0.0.1', port=5000) 