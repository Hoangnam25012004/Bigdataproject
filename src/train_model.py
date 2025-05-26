import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Attention, LayerNormalization, Input, Concatenate, GRU
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import re
import string
import emoji
import pickle
import sys
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

def discover_sentiment_words(df, min_freq=10, correlation_threshold=0.1, batch_size=1000):
    """
    Discover sentiment-related words from the dataset based on correlation with sentiment labels
    Uses batch processing and vectorized operations for better performance
    """
    print("\nDiscovering sentiment words...")
    
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
    
    print("Cleaning texts...")
    df['clean_text'] = df['tweet'].apply(clean_text)
    
    # Create document-term matrix with progress updates
    print("Creating document-term matrix...")
    vectorizer = CountVectorizer(min_df=min_freq)
    X = vectorizer.fit_transform(df['clean_text'])
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"Analyzing {len(feature_names)} unique words...")
    
    # Get word frequencies
    word_freq = pd.DataFrame(
        X.sum(axis=0).T,
        index=feature_names,
        columns=['frequency']
    )
    
    # Initialize sentiment words dictionary
    sentiment_words = {
        'positive': set(),
        'negative': set(),
        'neutral': set(),
        'intensifiers': set()
    }
    
    # Process sentiments in batches
    print("Calculating correlations...")
    for sentiment in df['sentiment'].unique():
        print(f"\nProcessing {sentiment} sentiment...")
        sentiment_docs = (df['sentiment'] == sentiment).astype(int)
        
        # Process words in batches
        for i in range(0, len(feature_names), batch_size):
            batch_end = min(i + batch_size, len(feature_names))
            batch_words = feature_names[i:batch_end]
            
            # Get document frequencies for batch words
            word_docs = X[:, i:batch_end].toarray()
            
            # Calculate correlations for the batch
            correlations = np.array([
                np.corrcoef(sentiment_docs, word_docs[:, j])[0, 1]
                for j in range(word_docs.shape[1])
            ])
            
            # Add words with significant correlations
            for word, corr in zip(batch_words, correlations):
                if not np.isnan(corr) and abs(corr) >= correlation_threshold:
                    if sentiment.lower() == 'positive' and corr > correlation_threshold:
                        sentiment_words['positive'].add(word)
                    elif sentiment.lower() == 'negative' and corr > correlation_threshold:
                        sentiment_words['negative'].add(word)
                    elif sentiment.lower() == 'neutral' and corr > correlation_threshold:
                        sentiment_words['neutral'].add(word)
    
    # Identify potential intensifiers
    print("\nIdentifying intensifiers...")
    for word in feature_names:
        if any(word.endswith(suffix) for suffix in ['ly', 'est', 'er']):
            sentiment_words['intensifiers'].add(word)
    
    # Print results
    print("\nDiscovered sentiment words:")
    for category, words in sentiment_words.items():
        print(f"\n{category.upper()} words (top 20):")
        print(", ".join(sorted(list(words))[:20]))
    
    return sentiment_words

def pre_process(text, sentiment_words):
    """
    Preprocess text using discovered sentiment words
    """
    if pd.isna(text) or not isinstance(text, str):
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

print("Loading data...")
try:
    # Load datasets with correct column names
    train_df = pd.read_csv('C:\\Users\\DELL\\Downloads\\bigdataproject\\twitter_training.csv',
                          names=['id', 'entity', 'sentiment', 'tweet'],
                          encoding='utf-8')
    val_df = pd.read_csv('C:\\Users\\DELL\\Downloads\\bigdataproject\\twitter_validation.csv',
                          names=['id', 'entity', 'sentiment', 'tweet'],
                          encoding='utf-8')
    
    print("Data loaded successfully!")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Combine datasets
    df = pd.concat([train_df, val_df])
    df = df[['sentiment', 'tweet']].dropna()
    
    # Print class distribution
    print("\nSentiment distribution:")
    print(df['sentiment'].value_counts())
    
    # Discover sentiment words from the dataset
    print("\nAnalyzing sentiment patterns in the data...")
    sentiment_words = discover_sentiment_words(df)
    
    # Clean the text using discovered patterns
    print("\nPreprocessing text...")
    df['clean_tweet'] = df['tweet'].apply(lambda x: pre_process(x, sentiment_words))
    
    # Remove empty tweets after preprocessing
    df = df[df['clean_tweet'] != ""]
    
    # Prepare text sequences
    print("\nTokenizing text...")
    tokenizer = Tokenizer(filters='', lower=True, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['clean_tweet'])
    
    sequences = tokenizer.texts_to_sequences(df['clean_tweet'])
    max_len = 50
    X = pad_sequences(sequences, maxlen=max_len, padding='post')
    
    # Prepare labels
    label_mapping = {'Positive': 0, 'Neutral': 1, 'Negative': 2, 'Irrelevant': 3}
    df['sentiment'] = df['sentiment'].map(label_mapping)
    df = df.dropna()
    y = pd.get_dummies(df['sentiment']).values
    
    # Calculate class weights
    y_integers = np.argmax(y, axis=1)
    class_weights_balanced = class_weight.compute_class_weight('balanced', 
                                                             classes=np.unique(y_integers), 
                                                             y=y_integers)
    class_weights = dict(enumerate(class_weights_balanced))
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    print("\nVocabulary size:", len(tokenizer.word_index))
    print("Maximum sequence length:", max_len)
    print("Number of classes:", len(label_mapping))
    
except Exception as e:
    print(f"Error loading or processing data: {str(e)}")
    sys.exit(1)

# Build model architecture
print("\nBuilding model...")
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 300

# Input layer
input_layer = Input(shape=(max_len,))
embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(input_layer)

# CNN branches with different kernel sizes
conv_blocks = []
for kernel_size in [3, 4, 5]:
    conv = Conv1D(128, kernel_size, activation='relu', padding='same')(embedding)
    conv = LayerNormalization()(conv)
    conv = MaxPooling1D(2)(conv)
    conv_blocks.append(conv)

# Combine CNN branches
cnn_features = Concatenate()(conv_blocks)

# Parallel LSTM and GRU branches
lstm_branch = Bidirectional(LSTM(256, return_sequences=True))(cnn_features)
lstm_branch = LayerNormalization()(lstm_branch)
lstm_branch = Dropout(0.5)(lstm_branch)

gru_branch = Bidirectional(GRU(256, return_sequences=True))(cnn_features)
gru_branch = LayerNormalization()(gru_branch)
gru_branch = Dropout(0.5)(gru_branch)

# Combine LSTM and GRU branches
combined = Concatenate()([lstm_branch, gru_branch])

# Additional sequence processing
seq_layer = Bidirectional(LSTM(128))(combined)
seq_layer = LayerNormalization()(seq_layer)
seq_layer = Dropout(0.4)(seq_layer)

# Dense layers with residual connections
dense1 = Dense(256, activation='relu')(seq_layer)
dense1 = LayerNormalization()(dense1)
dense1 = Dropout(0.5)(dense1)

dense2 = Dense(128, activation='relu')(dense1)
dense2 = LayerNormalization()(dense2)
dense2 = Dropout(0.4)(dense2)

# Residual connection
dense2_with_residual = Concatenate()([dense2, Dense(128, activation='relu')(seq_layer)])

output = Dense(4, activation='softmax')(dense2_with_residual)

# Create model
model = Model(inputs=input_layer, outputs=output)

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("\nModel summary:")
model.summary()

# Add callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6
)

print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weights,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f'\nTest Accuracy: {accuracy:.2f}')

# Save model and tokenizer
print("\nSaving model and tokenizer...")
model.save('sentiment_lstm_model.h5')

# Save tokenizer and discovered sentiment words
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('sentiment_words.pickle', 'wb') as handle:
    pickle.dump(sentiment_words, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("\nDone! You can now run sentiment_api.py") 