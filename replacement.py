import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
import json

# Dataset-klass (samma som i din ursprungskod)
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    return total_loss / len(train_loader), all_preds, all_labels

def evaluate(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(val_loader), all_preds, all_labels

def main():
    # Ange dataset-path här (ändra om du har annan sökväg)
    dataset_path = 'nya_datasetet.csv'

    # Sätt random seed för reproducerbarhet
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"✅ Använder GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    else:
        print("⚠️ Ingen GPU hittades, använder CPU")

    print("\nLaddar dataset...")
    df = pd.read_csv(dataset_path)
    print(f"✅ Laddat {len(df)} artiklar")

    # Ingen label-mappning behövs, dina labels är redan int och rätt!
    df = df[df['label'].isin([0, 1])].reset_index(drop=True)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].values, df['label'].values,
        test_size=0.2, random_state=42, stratify=df['label'].values
    )
    print(f"Träningsdata: {len(train_texts)} | Valideringsdata: {len(val_texts)}")

    model_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        hidden_dropout_prob=0.1,  # Sänkt från 0.3 till 0.1
        attention_probs_dropout_prob=0.1  # Sänkt från 0.3 till 0.1
    )
    model.to(device)

    train_dataset = NewsDataset(train_texts, train_labels, tokenizer)
    val_dataset = NewsDataset(val_texts, val_labels, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    optimizer = AdamW(
        model.parameters(),
        lr=3e-5,  # Sänkt från 5e-5 till 3e-5
        weight_decay=0.01,
        eps=1e-8
    )
    num_epochs = 6
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"trained_models/xlmroberta_model_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss, train_preds, train_labels = train_epoch(
            model, train_loader, optimizer, device
        )
        val_loss, val_preds, val_labels = evaluate(model, val_loader, device)
        print("\nTräningsmetriker:")
        print(classification_report(train_labels, train_preds))
        print("\nValideringsmetriker:")
        print(classification_report(val_labels, val_preds))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"\nSparar bättre modell (val_loss: {val_loss:.4f})")
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            results = classification_report(val_labels, val_preds, output_dict=True)
            results['val_loss'] = val_loss
            with open(f"{model_dir}/eval_results.json", 'w') as f:
                json.dump(results, f, indent=2)
    print("\nTräning avslutad!")
    print(f"Bästa modellen sparad i: {model_dir}")

if __name__ == "__main__":
    main() 