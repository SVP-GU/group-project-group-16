import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import torch
from torch.utils.data import Dataset
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
from datetime import datetime
import re
import nltk
from nltk.corpus import stopwords
import glob
from googletrans import Translator
from langdetect import detect
import time

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    """
    Beräknar utvärderingsmetriker för modellen.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    
    return {
        'accuracy': report['accuracy'],
        'f1_misinformation': report['0']['f1-score'],
        'f1_true': report['1']['f1-score'],
        'macro_f1': report['macro avg']['f1-score']
    }

def prepare_data():
    """
    Förbereder data för träning genom att läsa CSV och dela upp i train/test.
    """
    # Läs data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'crawled_articles.csv')
    df = pd.read_csv(csv_path)
    print(f"✅ Läste in: {len(df)} artiklar")
    
    # Konvertera text till strings och hantera NaN-värden
    df['text'] = df['text'].fillna('').astype(str)
    
    # Dela upp i train/test
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['text'].values, 
        df['label'].values, 
        test_size=0.2, 
        random_state=42
    )
    
    # Konvertera numpy arrays till listor för att säkerställa rätt format
    train_texts = train_texts.tolist()
    test_texts = test_texts.tolist()
    train_labels = train_labels.tolist()
    test_labels = test_labels.tolist()
    
    return train_texts, test_texts, train_labels, test_labels

def train_model():
    """
    Tränar en TF-IDF + LogisticRegression modell på det förbehandlade datasetet
    """
    # Läs in det förbehandlade datasetet
    print("Läser in förbehandlad data...")
    df = pd.read_csv('data/combined_dataset_preprocessed.csv')
    print(f"Läste in {len(df)} artiklar")
    
    # Dela upp i features och labels
    X = df['text']
    y = df['label']
    
    # Dela upp i tränings- och testdata
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # TF-IDF Vektorisering
    print("\nSkapar TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    # Träna modellen
    print("Tränar modellen...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vectorized, y_train)
    
    # Utvärdera modellen
    y_pred = model.predict(X_test_vectorized)
    
    # Skriv ut utvärderingsmetrik
    print("\nKlassificeringsrapport:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Skapa mapp för modeller om den inte finns
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Spara modellen och vektoriseraren
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = os.path.join(model_dir, f'model_{timestamp}.joblib')
    vectorizer_filename = os.path.join(model_dir, f'vectorizer_{timestamp}.joblib')
    
    joblib.dump(model, model_filename)
    joblib.dump(vectorizer, vectorizer_filename)
    
    print(f"\nModell sparad som: {model_filename}")
    print(f"Vektoriserare sparad som: {vectorizer_filename}")
    
    return model, vectorizer

def load_and_preprocess_data(file_path):
    """
    Laddar och förbehandlar data från CSV-filen
    """
    # Läs in data
    df = pd.read_csv(file_path)
    
    # Kombinera titel och text
    df['full_text'] = df['title'] + ' ' + df['text']
    
    # Grundläggande textförbehandling
    def preprocess_text(text):
        if isinstance(text, str):
            # Konvertera till lowercase
            text = text.lower()
            # Ta bort specialtecken och siffror
            text = re.sub(r'[^a-öA-Ö\s]', '', text)
            # Ta bort extra whitespace
            text = ' '.join(text.split())
            return text
        return ''

    # Applicera förbehandling
    df['processed_text'] = df['full_text'].apply(preprocess_text)
    
    return df

def train_model_tfidf(df, model_save_path='trained_model'):
    """
    Tränar en modell på den förbehandlade datan
    """
    # Förbereder features och labels
    X = df['processed_text']
    y = df['label']
    
    # Dela upp i tränings- och testdata
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # TF-IDF Vektorisering
    vectorizer = TfidfVectorizer(max_features=10000, 
                                stop_words=stopwords.words('swedish'),
                                ngram_range=(1, 2))
    
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    # Träna modellen
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vectorized, y_train)
    
    # Utvärdera modellen
    y_pred = model.predict(X_test_vectorized)
    
    # Skriv ut utvärderingsmetrik
    print("\nKlassificeringsrapport:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Spara modellen och vektoriseraren
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f'{model_save_path}_model_{timestamp}.joblib'
    vectorizer_filename = f'{model_save_path}_vectorizer_{timestamp}.joblib'
    
    joblib.dump(model, model_filename)
    joblib.dump(vectorizer, vectorizer_filename)
    
    print(f"\nModell sparad som: {model_filename}")
    print(f"Vektoriserare sparad som: {vectorizer_filename}")
    
    return model, vectorizer, (X_test, y_test, y_pred)

class NewsClassifier:
    def __init__(self, model_path=None, vectorizer_path=None):
        """
        Initierar klassificeraren med en tränad modell och vektoriserare
        """
        self.model = joblib.load(model_path) if model_path else None
        self.vectorizer = joblib.load(vectorizer_path) if vectorizer_path else None
        self.translator = Translator()
    
    def translate_to_swedish(self, text):
        """
        Översätter text från engelska till svenska
        """
        try:
            if not isinstance(text, str) or not text.strip():
                return ''
            
            # Dela upp texten i mindre bitar för att undvika översättningsgränser
            max_length = 5000
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            
            translated_chunks = []
            for chunk in chunks:
                try:
                    translated = self.translator.translate(chunk, dest='sv', src='en')
                    translated_chunks.append(translated.text)
                    # Lägg till en kort paus mellan översättningar
                    time.sleep(1)
                except Exception as e:
                    print(f"Varning: Kunde inte översätta chunk: {str(e)}")
                    translated_chunks.append(chunk)
            
            return ' '.join(translated_chunks)
        except Exception as e:
            print(f"Varning: Översättningsfel: {str(e)}")
            return text
    
    def preprocess_text(self, text, translate_if_english=True):
        """
        Förbehandlar text och översätter om den är på engelska
        """
        if not isinstance(text, str) or not text.strip():
            return ''
        
        # Identifiera språk
        try:
            lang = detect(text)
        except:
            lang = 'sv'  # Anta svenska som default
        
        # Översätt om texten är på engelska och översättning är aktiverad
        if lang == 'en' and translate_if_english:
            text = self.translate_to_swedish(text)
        
        return text
    
    def predict(self, text, translate_if_english=True):
        """
        Gör en prediktion på en text
        """
        if not self.model or not self.vectorizer:
            raise ValueError("Modellen är inte tränad eller laddad!")
        
        # Förbehandla texten
        processed_text = self.preprocess_text(text, translate_if_english)
        
        # Vektorisera texten
        text_vectorized = self.vectorizer.transform([processed_text])
        
        # Gör prediktion
        prediction = self.model.predict(text_vectorized)[0]
        probability = self.model.predict_proba(text_vectorized)[0]
        
        return {
            'prediction': prediction,
            'probability': probability.max(),
            'was_translated': processed_text != text
        }
    
    def train(self, preprocessed_data_path='data/combined_dataset_preprocessed.csv'):
        """
        Tränar modellen med förbättrade tekniker
        """
        print("Läser in förbehandlad data...")
        df = pd.read_csv(preprocessed_data_path)
        print(f"Läste in {len(df)} artiklar")
        
        # Dela upp i features och labels
        X = df['text']
        y = df['label']
        
        # Dela upp i tränings- och testdata
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # TF-IDF Vektorisering med optimerade parametrar
        print("\nSkapar TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            max_features=15000,  # Ökat antal features
            ngram_range=(1, 3),  # Inkludera 1-3 grams
            min_df=2,           # Ignorera termer som förekommer i mindre än 2 dokument
            max_df=0.95,        # Ignorera termer som förekommer i mer än 95% av dokumenten
            strip_accents='unicode',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True   # Använd logaritmisk term-frekvens
        )
        
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        # Använd SMOTE för att balansera datasetet
        print("Balanserar datasetet med SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_vectorized, y_train)
        
        # Beräkna class weights
        class_weights = dict(zip(
            np.unique(y_train),
            len(y_train) / (len(np.unique(y_train)) * np.bincount(y_train))
        ))
        
        # Grid Search för hyperparameter tuning
        print("Utför hyperparameter tuning...")
        param_grid = {
            'C': [0.1, 1, 10],
            'max_iter': [1000],
            'class_weight': ['balanced', class_weights],
            'solver': ['lbfgs', 'saga'],
            'penalty': ['l2']
        }
        
        grid_search = GridSearchCV(
            LogisticRegression(),
            param_grid,
            cv=5,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train_balanced, y_train_balanced)
        
        print("\nBästa parametrar:", grid_search.best_params_)
        self.model = grid_search.best_estimator_
        
        # Utvärdera på testdata
        y_pred = self.model.predict(X_test_vectorized)
        
        print("\nKlassificeringsrapport:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Spara modellen
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_models')
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = os.path.join(model_dir, f'model_improved_{timestamp}.joblib')
        vectorizer_filename = os.path.join(model_dir, f'vectorizer_improved_{timestamp}.joblib')
        
        joblib.dump(self.model, model_filename)
        joblib.dump(self.vectorizer, vectorizer_filename)
        
        print(f"\nModell sparad som: {model_filename}")
        print(f"Vektoriserare sparad som: {vectorizer_filename}")
        
        # Visa detaljerad utvärdering
        print("\nDetaljerad utvärdering:")
        print("\nPer-klass precision:")
        for i, label in enumerate(['Desinformation', 'Trovärdig']):
            precision = classification_report(y_test, y_pred, output_dict=True)['%d' % i]['precision']
            recall = classification_report(y_test, y_pred, output_dict=True)['%d' % i]['recall']
            f1 = classification_report(y_test, y_pred, output_dict=True)['%d' % i]['f1-score']
            print(f"{label}:")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1-score: {f1:.3f}")

def main():
    # Skapa och träna klassificeraren
    classifier = NewsClassifier()
    classifier.train()
    
    # Testa några exempel
    print("\nTestar klassificeraren på några exempel:")
    
    # Svenska exempel
    sv_text = """
    Sveriges regering meddelade idag att nya åtgärder införs för att bekämpa klimatförändringarna.
    Beslutet togs efter omfattande diskussioner med experter och miljöorganisationer.
    """
    
    # Engelska exempel
    en_text = """
    Breaking news: Scientists discover groundbreaking new treatment for cancer.
    The research, published in Nature, shows promising results in clinical trials.
    """
    
    print("\nTestar svensk text:")
    result_sv = classifier.predict(sv_text)
    print(f"Prediktion: {'Trovärdig' if result_sv['prediction'] == 1 else 'Desinformation'}")
    print(f"Sannolikhet: {result_sv['probability']:.2f}")
    
    print("\nTestar engelsk text:")
    result_en = classifier.predict(en_text)
    print(f"Prediktion: {'Trovärdig' if result_en['prediction'] == 1 else 'Desinformation'}")
    print(f"Sannolikhet: {result_en['probability']:.2f}")
    print(f"Texten översattes: {result_en['was_translated']}")

if __name__ == "__main__":
    print("Startar förbättrad modellträning och testning...")
    main() 