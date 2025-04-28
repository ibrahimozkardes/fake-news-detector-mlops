import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import os

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """
    Clean text by:
    - Converting to lowercase
    - Removing punctuation and special characters
    - Removing stopwords
    - Applying lemmatization
    """
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and apply lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    # Join tokens back into text
    return ' '.join(tokens)

def preprocess_data(file_path, output_path):
    """
    Clean data and save processed data without index column.
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data
    data = pd.read_csv(file_path)
    
    # Remove "Unnamed: 0" column if it exists
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
    
    # Clean text column
    data['cleaned_text'] = data['text'].apply(clean_text)
    
    # Save processed data without index
    data.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    # Use os.path.join for better path handling
    raw_data_path = os.path.join("data", "raw", "fake_or_real_news.csv")
    processed_data_path = os.path.join("data", "processed", "cleaned_fake_or_real_news.csv")
    
    preprocess_data(raw_data_path, processed_data_path)