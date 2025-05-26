import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
from collections import defaultdict
import logging
from tqdm import tqdm
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
from data_utils import (
    create_category_balanced_splits,
    create_cross_validation_splits,
    create_dataloaders,
    print_split_statistics
)

# Konfigurera logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsDataset(Dataset):
    def __init__(self, texts, labels, categories, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.categories = categories
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
            'category': self.categories[idx]
        }

class CategoryAwareNewsClassifier(nn.Module):
    def __init__(self, num_categories=6):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.category_embeddings = nn.Embedding(num_categories, 768)
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )
        
    def forward(self, input_ids, attention_mask, category):
        # BERT encoding
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_embedding = bert_output.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Category encoding
        category_embedding = self.category_embeddings(category)
        
        # Combine embeddings
        combined = torch.cat([text_embedding, category_embedding], dim=1)
        
        # Classification
        return self.classifier(combined)

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        category = batch['category'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, category)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(train_loader), accuracy_score(all_labels, all_preds)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_categories = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            category = batch['category'].to(device)
            
            outputs = model(input_ids, attention_mask, category)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_categories.extend(category.cpu().numpy())
    
    # Beräkna metrics per kategori
    category_metrics = {}
    for cat in range(6):  # 6 kategorier
        cat_mask = np.array(all_categories) == cat
        if sum(cat_mask) > 0:
            cat_preds = np.array(all_preds)[cat_mask]
            cat_labels = np.array(all_labels)[cat_mask]
            cat_acc = accuracy_score(cat_labels, cat_preds)
            cat_prec, cat_rec, cat_f1, _ = precision_recall_fscore_support(
                cat_labels, cat_preds, average='binary'
            )
            category_metrics[cat] = {
                'accuracy': cat_acc,
                'precision': cat_prec,
                'recall': cat_rec,
                'f1': cat_f1
            }
    
    return (
        total_loss / len(val_loader),
        accuracy_score(all_labels, all_preds),
        category_metrics,
        confusion_matrix(all_labels, all_preds)
    )

def plot_confusion_matrix(cm, fold=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix{" - Fold " + str(fold) if fold is not None else ""}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'model/new_model/visualizations/confusion_matrix{"_fold" + str(fold) if fold is not None else ""}.png')
    plt.close()

def plot_metrics(metrics_history, fold=None):
    plt.figure(figsize=(15, 5))
    
    # Plot training metrics
    plt.subplot(1, 2, 1)
    plt.plot(metrics_history['train_loss'], label='Training Loss')
    plt.plot(metrics_history['val_loss'], label='Validation Loss')
    plt.title(f'Loss Over Time{" - Fold " + str(fold) if fold is not None else ""}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(metrics_history['train_acc'], label='Training Accuracy')
    plt.plot(metrics_history['val_acc'], label='Validation Accuracy')
    plt.title(f'Accuracy Over Time{" - Fold " + str(fold) if fold is not None else ""}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'model/new_model/visualizations/metrics{"_fold" + str(fold) if fold is not None else ""}.png')
    plt.close()

def main():
    # Ladda data
    df = pd.read_csv('data/combined_dataset_preprocessed.csv')
    
    # Skapa kategorier baserat på textinnehåll
    def categorize_text(text):
        text = text.lower()
        if any(word in text for word in ['trump', 'biden', 'election', 'president', 'congress', 'senate']):
            return 0  # politics
        elif any(word in text for word in ['market', 'stock', 'economy', 'business', 'trade', 'dollar']):
            return 1  # economy
        elif any(word in text for word in ['covid', 'virus', 'health', 'disease', 'medical', 'hospital']):
            return 2  # health
        elif any(word in text for word in ['tech', 'technology', 'digital', 'software', 'computer', 'internet']):
            return 3  # technology
        elif any(word in text for word in ['movie', 'film', 'actor', 'actress', 'celebrity', 'entertainment']):
            return 4  # entertainment
        else:
            return 5  # sports
    
    df['category'] = df['text'].apply(categorize_text)
    
    # Skapa k-fold splits
    n_splits = 5
    splits = create_cross_validation_splits(df, n_splits=n_splits)
    
    # Träna modellen för varje fold
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    fold_metrics = []
    
    for fold, (train_df, val_df) in enumerate(splits):
        print(f"\nTraining Fold {fold + 1}/{n_splits}")
        print("=" * 50)
        
        # Skapa dataloaders
        train_loader, val_loader, _ = create_dataloaders(
            train_df, val_df, None, tokenizer
        )
        
        # Initiera modell och optimizer
        model = CategoryAwareNewsClassifier().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss()
        
        # Träna modellen
        best_val_acc = 0
        metrics_history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        
        for epoch in range(5):
            print(f"\nEpoch {epoch + 1}/5")
            
            # Träna
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device
            )
            
            # Evalvera
            val_loss, val_acc, category_metrics, cm = evaluate(
                model, val_loader, criterion, device
            )
            
            # Spara metrics
            metrics_history['train_loss'].append(train_loss)
            metrics_history['val_loss'].append(val_loss)
            metrics_history['train_acc'].append(train_acc)
            metrics_history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Spara bästa modell
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'model/new_model/saved_models/best_model_fold_{fold}.pt')
                
                # Spara confusion matrix
                plot_confusion_matrix(cm, fold)
        
        # Spara metrics plot
        plot_metrics(metrics_history, fold)
        
        # Spara fold metrics
        fold_metrics.append({
            'fold': fold,
            'best_val_acc': best_val_acc,
            'category_metrics': category_metrics
        })
    
    # Skriv ut sammanfattning av alla folds
    print("\nCross-Validation Results:")
    print("=" * 50)
    
    for fold_metric in fold_metrics:
        print(f"\nFold {fold_metric['fold'] + 1}:")
        print(f"Best Validation Accuracy: {fold_metric['best_val_acc']:.4f}")
        print("\nCategory Metrics:")
        for cat, metrics in fold_metric['category_metrics'].items():
            print(f"Category {cat}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")

if __name__ == "__main__":
    main() 