import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
from utils import load_data, split_data, evaluate_model

def train_and_save_model(data_path, model_output_path):
    """
    Train a fake news detection model and save it to a file.
    """
    # Load and split the data
    data = load_data(data_path)

    # Handle missing values in the cleaned_text column
    data = data.dropna(subset=['cleaned_text'])  # Drop rows with NaN in cleaned_text
    data['cleaned_text'] = data['cleaned_text'].fillna('')  # Alternatively, fill NaN with an empty string

    X_train, X_test, y_train, y_test = split_data(data, text_column='cleaned_text', label_column='label')

    # Create a pipeline with TF-IDF vectorizer and Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    evaluate_model(y_test, y_pred)

    # Save the trained model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(pipeline, model_output_path)
    print(f"Model saved to {model_output_path}")

if __name__ == "__main__":
    # Define paths
    processed_data_path = "data/processed/cleaned_fake_or_real_news.csv"
    model_output_path = "models/fake_news_detector.pkl"

    # Train and save the model
    train_and_save_model(processed_data_path, model_output_path)