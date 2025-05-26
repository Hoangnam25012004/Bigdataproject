import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, regexp_replace, lower, when
from pyspark.sql.types import StringType, ArrayType, FloatType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import re
import string
import emoji
import pickle
import nltk
from pathlib import Path

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def setup_hadoop_env():
    """Setup Hadoop environment for Windows"""
    hadoop_home = "C:\\hadoop"
    hadoop_bin = os.path.join(hadoop_home, 'bin')
    winutils_path = os.path.join(hadoop_bin, 'winutils.exe')
    hadoop_dll_path = os.path.join(hadoop_bin, 'hadoop.dll')
    
    # Create directories if they don't exist
    os.makedirs(hadoop_bin, exist_ok=True)
    os.makedirs('C:\\temp', exist_ok=True)
    
    # Check for required files
    files_missing = False
    if not os.path.exists(winutils_path):
        print(f"Missing {winutils_path}")
        files_missing = True
    if not os.path.exists(hadoop_dll_path):
        print(f"Missing {hadoop_dll_path}")
        files_missing = True
    
    if files_missing:
        print("\nPlease follow these steps to set up Hadoop environment:")
        print("1. Download winutils.exe and hadoop.dll from: https://github.com/cdarlint/winutils/tree/master/hadoop-3.2.0/bin")
        print(f"2. Create the directory: {hadoop_bin}")
        print(f"3. Place both files in: {hadoop_bin}")
        print("\nAfter placing the files, run this script again.")
        sys.exit(1)
    
    # Set environment variables
    os.environ['HADOOP_HOME'] = hadoop_home
    os.environ['SPARK_LOCAL_DIRS'] = 'C:\\temp'
    os.environ['PATH'] = os.environ['PATH'] + ';' + hadoop_bin
    
    # Add Python to PATH if not already there
    python_dir = os.path.dirname(sys.executable)
    if python_dir not in os.environ['PATH']:
        os.environ['PATH'] = python_dir + ';' + os.environ['PATH']
    
    # Set JAVA_HOME directly
    java_home = "C:\\Program Files\\Java\\jdk-17"
    if os.path.exists(java_home):
        os.environ['JAVA_HOME'] = java_home
        print(f"\nUsing Java installation at: {java_home}")
    else:
        print("\nError: JDK 17 not found at expected location.")
        print("Please install JDK 17 from: https://www.oracle.com/java/technologies/downloads/#java17")
        sys.exit(1)

def create_spark_session():
    """Create and configure Spark session"""
    try:
        # Get Python executable path
        python_path = sys.executable
        python_dir = os.path.dirname(python_path)
        
        # Add Python to system PATH
        if sys.platform.startswith('win'):
            os.environ['PATH'] = f"{python_dir};{python_dir}\\Scripts;{os.environ['PATH']}"
            os.environ['PYSPARK_PYTHON'] = python_path
            os.environ['PYSPARK_DRIVER_PYTHON'] = python_path
        
        # Create a minimal Spark session
        return (SparkSession.builder
            .appName("SentimentAnalysis")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.python.worker.reuse", "true")
            .config("spark.python.executable", python_path)
            .master("local[1]")  # Use single core for stability
            .getOrCreate())
    except Exception as e:
        print(f"Error creating Spark session: {str(e)}")
        sys.exit(1)

# Setup Hadoop environment
print("Setting up Hadoop environment...")
setup_hadoop_env()

# Create Spark session
print("Creating Spark session...")
spark = create_spark_session()

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

print("Loading data...")
try:
    # Load datasets
    train_df = spark.read.csv('twitter_training.csv',
                             header=False,
                             inferSchema=True)\
                        .toDF("id", "entity", "sentiment", "tweet")
    
    val_df = spark.read.csv('twitter_validation.csv',
                           header=False,
                           inferSchema=True)\
                      .toDF("id", "entity", "sentiment", "tweet")
    
    print("Data loaded successfully!")
    print(f"Training samples: {train_df.count()}")
    print(f"Validation samples: {val_df.count()}")
    
    # Combine datasets
    df = train_df.union(val_df)
    df = df.select("sentiment", "tweet").dropna()
    
    # Print class distribution
    print("\nSentiment distribution:")
    df.groupBy("sentiment").count().show()
    
    # Preprocess text
    print("\nPreprocessing text...")
    df = df.withColumn("clean_tweet", pre_process_udf(col("tweet")))
    
    # Create pipeline stages
    tokenizer = Tokenizer(inputCol="clean_tweet", outputCol="tokens")
    
    # Remove stopwords but keep negation words
    stopwords = nltk.corpus.stopwords.words("english")
    negation_words = {"no", "not", "nor", "none", "never", "neither"}
    stopwords = list(set(stopwords) - negation_words)
    
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens", stopWords=stopwords)
    
    # Create word vectors
    vectorizer = CountVectorizer(inputCol="filtered_tokens", outputCol="raw_features", minDF=5.0)
    idf = IDF(inputCol="raw_features", outputCol="features")
    
    # Initialize logistic regression
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="sentiment_index",  # We'll create this
        maxIter=20,
        regParam=0.3,
        elasticNetParam=0
    )
    
    # Split data
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
    
    # Create sentiment index
    sentiment_indexer = StringIndexer(
        inputCol="sentiment",
        outputCol="sentiment_index",
        handleInvalid="keep"
    )
    
    # Create pipeline
    pipeline = Pipeline(stages=[
        tokenizer,
        remover,
        vectorizer,
        idf,
        sentiment_indexer,
        lr
    ])
    
    # Train model
    print("\nTraining model...")
    model = pipeline.fit(train_data)
    
    # Make predictions on test data
    predictions = model.transform(test_data)
    
    # Evaluate model
    evaluator = MulticlassClassificationEvaluator(
        labelCol="sentiment_index",
        predictionCol="prediction",
        metricName="accuracy"
    )
    accuracy = evaluator.evaluate(predictions)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Save the model and vectorizer
    print("\nSaving model and vectorizer...")
    model_path = "spark_sentiment_model"
    model.write().overwrite().save(model_path)
    
    # Extract and save vocabulary for later use
    vectorizer_model = model.stages[2]
    vocab = vectorizer_model.vocabulary
    
    # Get label mapping from the indexer
    label_mapping = {
        float(idx): label
        for idx, label in enumerate(model.stages[4].labels)
    }
    
    # Print label mapping for verification
    print("\nLabel Mapping:")
    for idx, label in label_mapping.items():
        print(f"Index {int(idx)}: {label}")
    
    # Get IDF values
    idf_model = model.stages[3]
    idf_values = idf_model.idf.toArray()
    
    vocab_data = {
        'vocabulary': vocab,
        'idf_values': idf_values.tolist(),  # Convert numpy array to list
        'label_mapping': label_mapping
    }
    
    # Verify the vocabulary data
    print("\nVocabulary data summary:")
    print(f"Number of terms: {len(vocab)}")
    print(f"Number of labels: {len(label_mapping)}")
    print(f"IDF values shape: {len(idf_values)}")
    
    with open('spark_sentiment_vocab.pickle', 'wb') as handle:
        pickle.dump(vocab_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("\nDone! You can now run sentiment_api.py")
    
except Exception as e:
    print(f"Error: {str(e)}")
    spark.stop()
    sys.exit(1)

spark.stop() 