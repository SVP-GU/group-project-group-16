import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import os

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

def prepare_data():
    """
    Förbereder data för träning genom att läsa CSV och dela upp i train/test.
    """
    # Läs data
    df = pd.read_csv('crawled_articles.csv')
    
    # Dela upp i train/test
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['text'].values, 
        df['label'].values, 
        test_size=0.2, 
        random_state=42
    )
    
    return train_texts, test_texts, train_labels, test_labels

def train_model():
    """
    Tränar KB-BERT-modellen på nyhetsdata.
    """
    # Skapa mapp för att spara modellen
    if not os.path.exists('saved_model'):
        os.makedirs('saved_model')
    
    # Förbered data
    train_texts, test_texts, train_labels, test_labels = prepare_data()
    
    # Initiera tokenizer och modell
    model_name = "KB/bert-base-swedish-cased-v1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Skapa datasets
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer)
    test_dataset = NewsDataset(test_texts, test_labels, tokenizer)
    
    # Definiera träningsargument
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )
    
    # Initiera trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )
    
    # Träna modellen
    trainer.train()
    
    # Spara modell och tokenizer
    model.save_pretrained('saved_model')
    tokenizer.save_pretrained('saved_model')
    
    print("Modell tränad och sparad i 'saved_model' mappen")

if __name__ == "__main__":
    train_model() 