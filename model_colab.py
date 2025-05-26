import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.utils.data import Dataset
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')

# Mount Google Drive
drive.mount('/content/drive')

def setup_colab():
    """
    Förbereder Colab-miljön med nödvändiga installationer och nedladdningar.
    """
    print("🚀 Förbereder Colab-miljön...")
    
    # Kontrollera GPU-tillgänglighet
    if torch.cuda.is_available():
        print(f"✅ GPU tillgänglig: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Ingen GPU tillgänglig, använder CPU")
    
    # Ladda ner nödvändiga NLTK-data
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    print("✅ Miljöförberedelse klar!")

def combine_datasets(swedish_csv_path, english_csv_path):
    """
    Kombinerar svenskt och engelskt dataset till ett enda dataset.
    """
    print("📊 Läser in dataset...")
    
    # Läs in båda dataseten
    df_sv = pd.read_csv(swedish_csv_path)
    df_en = pd.read_csv(english_csv_path)
    
    print(f"✅ Läste in: {len(df_sv)} svenska artiklar")
    print(f"✅ Läste in: {len(df_en)} engelska artiklar")
    
    # Lägg till en kolumn som indikerar språk
    df_sv['language'] = 'sv'
    df_en['language'] = 'en'
    
    # Kombinera dataseten
    df_combined = pd.concat([df_sv, df_en], ignore_index=True)
    print(f"✅ Totalt antal artiklar efter kombination: {len(df_combined)}")
    
    # Visa dataset-statistik
    print("\n📈 Dataset-statistik:")
    print("\nSpråkfördelning:")
    print(df_combined['language'].value_counts())
    print("\nKlassfördelning:")
    print(df_combined['label'].value_counts())
    
    return df_combined

def visualize_class_distribution(df):
    """
    Visualiserar klassfördelningen i datasetet.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='label')
    plt.title('Fördelning av klasser i datasetet')
    plt.xlabel('Klass (0=Desinformation, 1=Trovärdig)')
    plt.ylabel('Antal artiklar')
    plt.show()

def train_model_colab(df, model_save_path='/content/drive/MyDrive/trained_models'):
    """
    Tränar modellen med Colab-optimeringar och visualiseringar.
    """
    print("🤖 Startar modellträning...")
    
    # Skapa mapp för modeller om den inte finns
    os.makedirs(model_save_path, exist_ok=True)
    
    # Dela upp i features och labels
    X = df['text']
    y = df['label']
    
    # Dela upp i tränings- och testdata
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # TF-IDF Vektorisering
    print("\n📝 Skapar TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        strip_accents='unicode',
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True
    )
    
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    # Använd SMOTE för att balansera datasetet
    print("⚖️ Balanserar datasetet med SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_vectorized, y_train)
    
    # Grid Search för hyperparameter tuning
    print("🔍 Utför hyperparameter tuning...")
    param_grid = {
        'C': [0.1, 1, 10],
        'max_iter': [1000],
        'class_weight': ['balanced'],
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
    
    print("\n✨ Bästa parametrar:", grid_search.best_params_)
    model = grid_search.best_estimator_
    
    # Utvärdera på testdata
    y_pred = model.predict(X_test_vectorized)
    
    # Visualisera resultat
    print("\n📊 Utvärderingsresultat:")
    print("\nKlassificeringsrapport:")
    print(classification_report(y_test, y_pred))
    
    # Rita förvirringsmatris
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Förvirringsmatris')
    plt.xlabel('Predikterad klass')
    plt.ylabel('Sann klass')
    plt.show()
    
    # Spara modellen
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = os.path.join(model_save_path, f'model_{timestamp}.joblib')
    vectorizer_filename = os.path.join(model_save_path, f'vectorizer_{timestamp}.joblib')
    
    joblib.dump(model, model_filename)
    joblib.dump(vectorizer, vectorizer_filename)
    
    print(f"\n💾 Modell sparad som: {model_filename}")
    print(f"💾 Vektoriserare sparad som: {vectorizer_filename}")
    
    return model, vectorizer, (X_test, y_test, y_pred)

def main_colab():
    """
    Huvudfunktion för Colab-miljön.
    """
    # Förbered miljön
    setup_colab()
    
    # Sökvägar till dataseten (anpassa efter din Google Drive-struktur)
    swedish_csv_path = '/content/drive/MyDrive/cleaned_articles_sv.csv'
    english_csv_path = '/content/drive/MyDrive/cleaned_articles_en.csv'
    
    # Kombinera dataseten
    df = combine_datasets(swedish_csv_path, english_csv_path)
    
    # Visualisera klassfördelning
    visualize_class_distribution(df)
    
    # Träna modellen
    model, vectorizer, (X_test, y_test, y_pred) = train_model_colab(df)
    
    # Testa några exempel
    print("\n🧪 Testar klassificeraren på några exempel:")
    
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
    
    # Skapa klassificerare
    classifier = NewsClassifier(model_path=model, vectorizer_path=vectorizer)
    
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
    print("🚀 Startar förbättrad modellträning i Colab...")
    main_colab() 