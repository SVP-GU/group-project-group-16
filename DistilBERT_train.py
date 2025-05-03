import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import numpy as np
from sklearn.metrics import classification_report

# Kolla CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Träning sker på: {device}")

# Läs in dataset
df = pd.read_csv('data/combined_dataset_extended.csv')
print(f"✅ Läste in: {len(df)} rader")

# Visa unika etiketter
print(f"Unika etiketter före mappning: {df['label'].unique()}")

# Skapa mappning
label_map = {
    'fake': 0, 'false': 0, 'FAKE': 0, 'FALSE': 0, 0: 0,
    'real': 1, 'true': 1, 'REAL': 1, 'TRUE': 1, 'trustworthy': 1, 1: 1
}

# Mappa etiketter
df['label'] = df['label'].map(label_map)

# Ta bort rader som fortfarande har NaN efter mappning
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

print(f"✅ Antal rader efter mappning och rensning: {len(df)}")

# Konvertera till Hugging Face Dataset
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# Ladda tokenizer och modell
checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2).to(device)

# Tokenisera datasetet
def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding=True)

tokenized_dataset = dataset.map(tokenize, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Utvärderingsfunktion
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    return {
        'accuracy': report['accuracy'],
        'f1_misinformation': report['0']['f1-score'],
        'f1_true': report['1']['f1-score'],
        'macro_f1': report['macro avg']['f1-score']
    }

# Träningsargument
training_args = TrainingArguments(
    output_dir="./bert_results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,  # 10 epoker
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    no_cuda=not torch.cuda.is_available()
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Starta träning
trainer.train()

# Spara tränad modell
model.save_pretrained('model/bert_model')
tokenizer.save_pretrained('model/bert_model')

print("✅ Träningen är klar och modellen sparad i 'model/bert_model'")
