import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Ladda modellen och tokenizern (från DistilBERT_train.py)
model_path = 'model/bert_model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Kolla device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"✅ Modell laddad på: {device}")

# Exempeltexter att testa
texts = [
    "The sky is blue.",
    "COVID-19 was created as a bioweapon.",
    "Vaccines cause autism.",
    "Water boils at 100 degrees Celsius.",
    "The moon landing was faked by NASA."
]

# Gör prediktioner
for text in texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        label = "misinformation" if predicted_class == 1 else "true"
        print(f"Text: {text}\nPrediction: {predicted_class} → {label}\n{'-'*50}")
