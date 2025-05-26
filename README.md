[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/rjgbNS53)
Arbetsgång
1. Datainsamling från EUvsDisinfo. Kanske komplettera med fler sidor som även har sann fakta.
2. Förbehandla text
3. Träna ML-Modell
4. Skapa en API-server för användning av extension via HTTP-anrop
5. Bygga extension

# Swedish News Disinformation Detector

<<<<<<< HEAD
## Project Overview
This project implements a machine learning model for detecting disinformation in Swedish news articles. The model can analyze both Swedish and English text (with automatic translation) to determine if an article is likely to contain disinformation.

=======
## 🚀 Snabbstart (Quick Start)

### 1. Skapa en ny Python-fil
Skapa en ny fil (t.ex. `check_article.py`) och kopiera in följande kod:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def check_news(text):
    # Ladda modell och tokenizer
    model = AutoModelForSequenceClassification.from_pretrained("Mirac1999/swedish-news-classifier")
    tokenizer = AutoTokenizer.from_pretrained("Mirac1999/swedish-news-classifier")
    
    # Förbered texten
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    # Gör prediktion
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    return {
        "is_trustworthy": prediction == 1,
        "confidence": confidence
    }

# Exempel på användning
if __name__ == "__main__":
    # Testa med en svensk artikel
    svensk_text = """
    Sveriges regering meddelade idag nya åtgärder för att bekämpa klimatförändringarna.
    Beslutet togs efter omfattande diskussioner med experter och miljöorganisationer.
    """
    
    result = check_news(svensk_text)
    print("\nTest med svensk text:")
    print(f"Är texten trovärdig? {'Ja' if result['is_trustworthy'] else 'Nej'}")
    print(f"Konfidens: {result['confidence']:.2%}")
    
    # Testa med en engelsk artikel (översätts automatiskt)
    english_text = """
    Breaking news: Scientists discover groundbreaking climate change solution.
    The new method could reverse global warming within years.
    """
    
    result = check_news(english_text)
    print("\nTest med engelsk text:")
    print(f"Är texten trovärdig? {'Ja' if result['is_trustworthy'] else 'Nej'}")
    print(f"Konfidens: {result['confidence']:.2%}")
```

### 2. Installera nödvändiga paket
Öppna en terminal och kör:
```bash
pip install transformers torch
```

### 3. Kör programmet
```bash
python check_article.py
```

Det är allt! Modellen kommer automatiskt att laddas ner första gången du kör programmet.

## 📊 Modellens Prestanda
- Accuracy: 77.1%
- Precision för trovärdiga nyheter: 88.5%
- Recall för vilseledande nyheter: 85.6%

## 🛠️ Felsökning

Om du får problem, kontrollera följande:

1. **ModuleNotFoundError**: Kör `pip install transformers torch` igen
2. **CUDA-fel**: Om du har problem med GPU, lägg till denna kod före modell-laddningen:
   ```python
   import torch
   device = "cuda" if torch.cuda.is_available() else "cpu"
   model = model.to(device)
   ```
3. **Minnesfel**: Om du får minnesfel, minska `max_length` i tokenizer-anropet

## 🔍 Tekniska Detaljer

### Modellarkitektur
- Basmodell: KB/bert-base-swedish-cased (Swedish BERT)
- Finjusterad på vår egen dataset av svenska nyhetsartiklar
- Stöd för både svenska och engelska texter

### Dataset
Modellen är tränad på 38,317 svenska nyhetsartiklar, inklusive både legitima nyheter och kända desinformationskällor.

## 📝 Mer Information

## Project Overview
This project implements a machine learning model for detecting disinformation in Swedish news articles. The model can analyze both Swedish and English text (with automatic translation) to determine if an article is likely to contain disinformation.

## Latest Model Performance (BERT)
Our latest BERT model achieves significantly improved performance:
- Overall Accuracy: 77.1%
- Trustworthy News:
  - Precision: 88.5%
- Misleading News:
  - Recall: 85.6%

## Technical Details

### Model Architecture
- Base Model: KB/bert-base-swedish-cased (Swedish BERT)
- Fine-tuned on our custom dataset of Swedish news articles
- Supports both Swedish and English text input (with automatic translation)

### Dataset
The model is trained on a comprehensive dataset of Swedish news articles, including both legitimate news and known disinformation sources.

## Previous Models
The repository also includes our previous machine learning models based on TF-IDF and traditional ML approaches. These can be found in the legacy code.

>>>>>>> d3e4ad8f0d38f246b30f63086fbe26a6bbd9ac2a
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
