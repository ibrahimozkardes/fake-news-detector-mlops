import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def split_data(data, text_column, label_column, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    """
    X = data[text_column]
    y = data[label_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def evaluate_model(y_true, y_pred):
    """
    Evaluate the model's performance and print metrics.
    """
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))