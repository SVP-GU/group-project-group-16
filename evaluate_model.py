import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from sklearn.metrics import classification_report
import torch

# ✅ Steg 1: Läs in data
df = pd.read_csv("data/combined_dataset_extended.csv")

# ✅ Steg 2: Mappa etiketter till 0/1
df['label'] = df['label'].replace({
    'true': 0,
    'false': 1,
    'real': 0,
    'fake': 1,
    'truthful': 0,
    'deceptive': 1,
    'trustworthy': 0,
    'untrustworthy': 1
})

# ✅ Steg 3: Ta bort rader som fortfarande har ogiltiga etiketter
df = df[df['label'].isin([0, 1])].reset_index(drop=True)

# ✅ Steg 4: Omvandla till Dataset-objekt
dataset = Dataset.from_pandas(df[['text', 'label']])
dataset = dataset.train_test_split(test_size=0.2, seed=42)
test_dataset = dataset['test']

# ✅ Steg 5: Ladda tokenizer och modell
tokenizer = AutoTokenizer.from_pretrained("model/bert_model")
model = AutoModelForSequenceClassification.from_pretrained("model/bert_model")

# ✅ Steg 6: Tokenisera testdatan
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

test_dataset = test_dataset.map(tokenize, batched=True)

# ✅ Steg 7: Gör prediktioner
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

predictions = []
labels = []

with torch.no_grad():
    for item in test_dataset:
        inputs = tokenizer(item['text'], return_tensors='pt', truncation=True, padding=True).to(device)
        output = model(**inputs)
        pred = torch.argmax(output.logits, dim=1).item()
        predictions.append(pred)
        labels.append(item['label'])

# ✅ Steg 8: Utvärdera
print("📊 Klassificeringsrapport:\n")
print(classification_report(labels, predictions, target_names=["true", "misinformation"]))
