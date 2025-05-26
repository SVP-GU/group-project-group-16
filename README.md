[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/rjgbNS53)
Arbetsgång
1. Datainsamling från EUvsDisinfo. Kanske komplettera med fler sidor som även har sann fakta.
2. Förbehandla text
3. Träna ML-Modell
4. Skapa en API-server för användning av extension via HTTP-anrop
5. Bygga extension

# Swedish News Disinformation Detector

Detta projekt är ett grupparbete som utvecklar en Streamlit-app för att klassificera svenska nyhetsartiklar som trovärdiga eller vilseledande, med hjälp av en fintränad RoBERTa-modell.

## 🚀 Snabbstart

### 1. Klona repot
```sh
git clone https://github.com/SVP-GU/group-project-group-16.git
cd group-project-group-16
```

### 2. Installera beroenden
Vi rekommenderar att använda en virtuell miljö:
```sh
pip install -r requirements.txt
```

### 3. Starta appen
```sh
streamlit run app.py --browser.gatherUsageStats false
```
Appen öppnas i din webbläsare på [http://localhost:8501](http://localhost:8501).

## 📰 Funktioner
- Klistra in en nyhetsartikel och analysera dess trovärdighet.
- Bygger på Hugging Face-modellen [`Mirac1999/roberta-new-classifier-2.0`](https://huggingface.co/Mirac1999/roberta-new-classifier-2.0).
- Resultat visas direkt i webbläsaren.

## 📊 Modell och Prestanda
- **Modell:** RoBERTa (XLM-RoBERTa-base) fintränad på svenska nyhetsartiklar
- **Valideringsresultat:**
  - Accuracy: 0.76
  - Macro F1: 0.76
  - Klass 0 (trovärdig): Precision 0.83, Recall 0.77, F1 0.80
  - Klass 1 (misinformation): Precision 0.68, Recall 0.75, F1 0.72

## 📚 Dataset
Modellen är tränad på ett sammansatt dataset av över 38 000 svenska nyhetsartiklar, hämtade från både legitima nyhetskällor och kända desinformationskällor (t.ex. EUvsDisinfo). Datasetet har förbehandlats och balanserats för att förbättra modellens förmåga att särskilja mellan trovärdig och vilseledande information.

## 🛠️ Exempel på användning i kod
Vill du använda modellen direkt i Python?
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("Mirac1999/roberta-new-classifier-2.0")
model = AutoModelForSequenceClassification.from_pretrained("Mirac1999/roberta-new-classifier-2.0")

text = "Sveriges regering meddelade idag nya åtgärder för att bekämpa klimatförändringarna."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][prediction].item()

print(f"Trovärdig: {prediction == 1}, Konfidens: {confidence:.2%}")
```

## 🤝 Bidra
Pull requests och förbättringsförslag välkomnas! Projektet är ett grupparbete inom kursen TIG321 på Göteborgs universitet.

## 📄 Licens
Se LICENSE-filen i repot.

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
