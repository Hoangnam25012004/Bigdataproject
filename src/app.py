# app.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
import re
import string
import emoji
import pickle
import numpy as np

def discover_sentiment_words(df, min_freq=10, correlation_threshold=0.1):
    """
    Discover sentiment-related words from the dataset based on correlation with sentiment labels
    """
    # Basic text cleaning for word discovery
    def clean_text(text):
        if pd.isna(text) or not isinstance(text, str):
            return ""
        text = emoji.demojize(text)
        text = text.lower()
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.strip()
    
    # Clean texts
    df['clean_text'] = df[text_column].apply(clean_text)
    
    # Create document-term matrix
    vectorizer = CountVectorizer(min_df=min_freq)
    X = vectorizer.fit_transform(df['clean_text'])
    
    # Get word frequencies
    word_freq = pd.DataFrame(
        X.sum(axis=0).T,
        index=vectorizer.get_feature_names_out(),
        columns=['frequency']
    )
    
    # Calculate correlation with sentiment
    sentiment_correlations = defaultdict(dict)
    feature_names = vectorizer.get_feature_names_out()
    
    for sentiment in df[sentiment_column].unique():
        sentiment_docs = (df[sentiment_column] == sentiment).astype(int)
        for idx, word in enumerate(feature_names):
            word_docs = X.getcol(idx).toarray().ravel()
            correlation = np.corrcoef(sentiment_docs, word_docs)[0, 1]
            if not np.isnan(correlation) and abs(correlation) >= correlation_threshold:
                sentiment_correlations[sentiment][word] = correlation
    
    # Organize words by sentiment
    sentiment_words = {
        'positive': set(),
        'negative': set(),
        'neutral': set(),
        'intensifiers': set()
    }
    
    # Words that strongly correlate with specific sentiments
    for sentiment, correlations in sentiment_correlations.items():
        for word, corr in correlations.items():
            if sentiment.lower() == 'positive' and corr > correlation_threshold:
                sentiment_words['positive'].add(word)
            elif sentiment.lower() == 'negative' and corr > correlation_threshold:
                sentiment_words['negative'].add(word)
            elif sentiment.lower() == 'neutral' and corr > correlation_threshold:
                sentiment_words['neutral'].add(word)
    
    # Identify potential intensifiers (words that appear frequently with strong sentiments)
    for word in feature_names:
        if any(word.endswith(suffix) for suffix in ['ly', 'est', 'er']):
            sentiment_words['intensifiers'].add(word)
    
    print("\nDiscovered sentiment words:")
    for category, words in sentiment_words.items():
        print(f"\n{category.upper()} words (top 20):")
        print(", ".join(sorted(list(words))[:20]))
    
    return sentiment_words, vectorizer

# Load data and print columns
print("Loading CSV files...")
train_df = pd.read_csv('twitter_training.csv')
val_df = pd.read_csv('twitter_validation.csv')

print("\nTraining CSV columns:", train_df.columns.tolist())
print("Validation CSV columns:", val_df.columns.tolist())

# Find text and sentiment columns
text_column = None
sentiment_column = None

possible_text_columns = ['text', 'tweet', 'message', 'content', 'Text', 'Tweet']
possible_sentiment_columns = ['sentiment', 'label', 'class', 'Sentiment', 'Label']

for col in train_df.columns:
    if col in possible_text_columns:
        text_column = col
    elif col in possible_sentiment_columns:
        sentiment_column = col

if text_column is None or sentiment_column is None:
    print("\nAvailable columns in training data:", train_df.columns.tolist())
    raise ValueError("Could not find text or sentiment columns. Please check your CSV files.")

print(f"\nUsing columns: {text_column} (for text) and {sentiment_column} (for sentiment)")

# Combine datasets
df = pd.concat([train_df, val_df])

# Discover sentiment words and get vectorizer
print("\nAnalyzing sentiment patterns in the data...")
sentiment_words, vectorizer = discover_sentiment_words(df)

# Save sentiment words for use by other components
with open('sentiment_words.pickle', 'wb') as handle:
    pickle.dump(sentiment_words, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Prepare training data
X_train = vectorizer.transform(train_df[text_column])
y_train = train_df[sentiment_column]

# Train model with class weights
class_weights = dict(zip(
    y_train.unique(),
    len(y_train) / (len(y_train.unique()) * np.bincount(y_train.factorize()[0]))
))

model = LogisticRegression(max_iter=1000, class_weight=class_weights)
model.fit(X_train, y_train)

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('sentiment_analysis.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
            
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400

        # Check in training set
        match = train_df[train_df[text_column].str.lower() == text.lower()]
        if not match.empty:
            sentiment = match[sentiment_column].values[0]
            return jsonify({
                'sentiment': sentiment,
                'confidence': 1.0,
                'source': 'training_data'
            })

        # Check in validation set
        match = val_df[val_df[text_column].str.lower() == text.lower()]
        if not match.empty:
            sentiment = match[sentiment_column].values[0]
            return jsonify({
                'sentiment': sentiment,
                'confidence': 1.0,
                'source': 'validation_data'
            })

        # If not found, use model prediction
        X_input = vectorizer.transform([text])
        pred = model.predict(X_input)[0]
        proba = model.predict_proba(X_input).max()
        
        # Add debug information about discovered sentiment words
        words = text.lower().split()
        debug_info = {
            'detected_negative_words': [w for w in words if w in sentiment_words['negative']],
            'detected_positive_words': [w for w in words if w in sentiment_words['positive']],
            'detected_neutral_words': [w for w in words if w in sentiment_words['neutral']],
            'detected_intensifiers': [w for w in words if w in sentiment_words['intensifiers']]
        }
        
        return jsonify({
            'sentiment': pred,
            'confidence': float(proba),
            'source': 'model_prediction',
            'debug_info': debug_info
        })
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
