# app.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Load data and print columns
print("Loading CSV files...")
train_df = pd.read_csv('twitter_training.csv')
val_df = pd.read_csv('twitter_validation.csv')

print("\nTraining CSV columns:", train_df.columns.tolist())
print("Validation CSV columns:", val_df.columns.tolist())

# Assuming the text column might have a different name, let's find it
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

# Combine for vectorizer fit
all_texts = pd.concat([train_df[text_column], val_df[text_column]])
vectorizer = CountVectorizer()
vectorizer.fit(all_texts)

# Prepare training data
X_train = vectorizer.transform(train_df[text_column])
y_train = train_df[sentiment_column]

# Train model
model = LogisticRegression(max_iter=1000)
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
        
        return jsonify({
            'sentiment': pred,
            'confidence': float(proba),
            'source': 'model_prediction'
        })
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
