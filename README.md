[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/rjgbNS53)
Arbetsgång
1. Datainsamling från EUvsDisinfo. Kanske komplettera med fler sidor som även har sann fakta.
2. Förbehandla text
3. Träna ML-Modell
4. Skapa en API-server för användning av extension via HTTP-anrop
5. Bygga extension

# Swedish News Disinformation Detector

## Project Overview
This project implements a machine learning model for detecting disinformation in Swedish news articles. The model can analyze both Swedish and English text (with automatic translation) to determine if an article is likely to contain disinformation.

## Recent Improvements

### 1. Enhanced Model Performance
- Increased accuracy to 74% with better balanced performance between classes
- Improved disinformation detection:
  - Precision: 82.1%
  - Recall: 70.2%
  - F1-score: 75.7%
- Better performance on trusted sources:
  - Precision: 66.4%
  - Recall: 79.3%
  - F1-score: 72.3%

### 2. Technical Improvements
- **Dataset Balancing**: Implemented SMOTE (Synthetic Minority Over-sampling Technique)
- **Hyperparameter Optimization**: Used GridSearchCV to find optimal parameters
- **Enhanced Feature Engineering**:
  - Increased TF-IDF features to 15,000
  - Extended n-gram range (1-3 words)
  - Implemented sublinear term frequency scaling
- **Multilingual Support**:
  - Automatic language detection
  - English-to-Swedish translation
  - Handles both languages seamlessly

### 3. Code Structure
- Implemented `NewsClassifier` class for better code organization
- Added comprehensive error handling
- Improved logging and progress reporting
- Better memory management for large datasets

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from model import NewsClassifier

# Initialize the classifier
classifier = NewsClassifier()

# Train the model
classifier.train()

# Make predictions
text = "Your news article text here..."
result = classifier.predict(text)
print(f"Prediction: {'Disinformation' if result['prediction'] == 0 else 'Trustworthy'}")
print(f"Confidence: {result['probability']:.2f}")
```

## Dataset
The model is trained on a comprehensive dataset of 38,317 Swedish news articles, including both legitimate news and known disinformation sources.

## Dependencies
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- googletrans
- langdetect
- joblib

## Future Improvements
- Implementation of deep learning models
- Real-time web scraping capabilities
- API endpoint for easy integration
- Browser extension for automatic article checking
