import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import DistilBertTokenizer
import torch
from collections import defaultdict

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

def create_category_balanced_splits(df, test_size=0.2, val_size=0.2, random_state=42):
    """
    Skapar balanserade train-val-test splits med stratifiering per kategori.
    """
    # Först dela upp i train+val och test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['category'],
        random_state=random_state
    )
    
    # Sedan dela train+val i train och val
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size/(1-test_size),  # Justera val_size för att kompensera för test_size
        stratify=train_val_df['category'],
        random_state=random_state
    )
    
    return train_df, val_df, test_df

def create_cross_validation_splits(df, n_splits=5, random_state=42):
    """
    Skapar k-fold cross-validation splits med stratifiering per kategori.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []
    
    for train_idx, val_idx in kf.split(df):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        splits.append((train_df, val_df))
    
    return splits

def create_dataloaders(train_df, val_df, test_df, tokenizer, batch_size=16):
    """
    Skapar DataLoaders för train, val och test med viktad sampling.
    """
    # Skapa datasets
    train_dataset = NewsDataset(
        train_df['text'].values,
        train_df['label'].values,
        train_df['category'].values,
        tokenizer
    )
    
    val_dataset = NewsDataset(
        val_df['text'].values,
        val_df['label'].values,
        val_df['category'].values,
        tokenizer
    )
    
    # Skapa test dataset endast om test_df finns
    test_loader = None
    if test_df is not None:
        test_dataset = NewsDataset(
            test_df['text'].values,
            test_df['label'].values,
            test_df['category'].values,
            tokenizer
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size*2,  # Större batch size för test
            shuffle=False
        )
    
    # Definiera kategori-vikter med numeriska nycklar
    category_weights = {
        0: 1.0,  # politics
        1: 2.0,  # economy
        2: 2.0,  # health
        3: 1.3,  # technology
        4: 1.0,  # entertainment
        5: 1.2   # sports
    }
    
    # Skapa viktade samplers för träning
    sample_weights = [
        category_weights[cat] for cat in train_df['category']
    ]
    sampler = WeightedRandomSampler(
        sample_weights,
        len(sample_weights),
        replacement=True
    )
    
    # Skapa dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size*2,  # Större batch size för validering
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader

def print_split_statistics(train_df, val_df, test_df):
    """
    Skriver ut statistik för varje split.
    """
    print("\nDataset Split Statistics:")
    print("=" * 50)
    
    for name, df in [("Training", train_df), ("Validation", val_df), ("Test", test_df)]:
        print(f"\n{name} Set:")
        print(f"Total samples: {len(df)}")
        
        # Label distribution
        label_dist = df['label'].value_counts()
        print("\nLabel distribution:")
        print(f"True news (0): {label_dist.get(0, 0)} ({label_dist.get(0, 0)/len(df)*100:.1f}%)")
        print(f"Fake news (1): {label_dist.get(1, 0)} ({label_dist.get(1, 0)/len(df)*100:.1f}%)")
        
        # Category distribution
        print("\nCategory distribution:")
        for category, count in df['category'].value_counts().items():
            print(f"{category}: {count} ({count/len(df)*100:.1f}%)") 