# Fake News Detector ML Project

## Project Overview
This project implements a machine learning pipeline for fake news detection using text classification techniques. The model analyzes news article content to classify whether articles are "REAL" or "FAKE" news.

## Model Performance
The trained model has achieved **90% accuracy** on the test dataset. This demonstrates the effectiveness of our approach combining TF-IDF vectorization with Logistic Regression.

## Project Structure
```
fake-news-detector-mlops/
├── data/
│   ├── processed/
│   │   └── cleaned_fake_or_real_news.csv
│   └── raw/
│       └── fake_or_real_news.csv
├── models/
│   └── fake_news_detector.pkl
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── utils.py
├── tests/
└── requirements.txt
```

## Model Training
The model was trained using the following pipeline:
1. **Data Preprocessing**: Text cleaning and normalization
2. **Feature Extraction**: TF-IDF Vectorization with max_features=5000
3. **Classification**: Logistic Regression with max_iter=1000

## Next Steps
- Experiment with different models (e.g., Random Forest, BERT)
- Implement cross-validation to ensure model robustness
- Add feature importance analysis to understand key predictors
- Create a web API for real-time predictions

## Getting Started
To train the model:
```bash
python src/train_model.py
```

To use the trained model for predictions:
```python
import joblib

model = joblib.load('models/fake_news_detector.pkl')

predictions = model.predict(['Text of news article to classify'])
```