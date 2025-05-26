import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import re
import string
import emoji
import pickle

def pre_process(text):
    if pd.isna(text) or not isinstance(text, str):
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

print("Loading data...")
# Load the CSV files
train_df = pd.read_csv('twitter_training.csv', names=['id', 'entity', 'sentiment', 'tweet'])
val_df = pd.read_csv('twitter_validation.csv', names=['id', 'entity', 'sentiment', 'tweet'])

# Combine datasets
df = pd.concat([train_df, val_df])

# Select only sentiment and tweet columns and remove rows with missing values
df = df[['sentiment', 'tweet']].dropna()

# Clean the text
print("Preprocessing text...")
df['clean_tweet'] = df['tweet'].apply(pre_process)

# Remove empty tweets after preprocessing
df = df[df['clean_tweet'] != ""]

# Prepare text sequences
print("Tokenizing text...")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['clean_tweet'])

sequences = tokenizer.texts_to_sequences(df['clean_tweet'])
max_len = 30
X = pad_sequences(sequences, maxlen=max_len, padding='post')

# Prepare labels
label_mapping = {'Positive': 0, 'Neutral': 1, 'Negative': 2, 'Irrelevant': 3}
df['sentiment'] = df['sentiment'].map(label_mapping)
df = df.dropna()  # Remove any rows where sentiment mapping failed
y = pd.get_dummies(df['sentiment']).values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Build the model
print("Building model...")
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100

model = Sequential([
    Embedding(vocab_size, embedding_dim),
    Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
print("Training model...")
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.2f}')

# Save the model and tokenizer
print("Saving model and tokenizer...")
model.save('sentiment_lstm_model.h5')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Done! You can now run sentiment_api.py") 