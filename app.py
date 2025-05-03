from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

app = Flask(__name__)
CORS(app)

# Bestäm device: kör på CPU för att undvika CUDA-problem i Flask
device = torch.device('cpu')

# Ladda modellen och tokenizern, och flytta modellen till rätt device
model_path = 'model/bert_model'
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'Ingen text skickades'}), 400

    # Förbered input och flytta till rätt device
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Gör förutsägelse
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()

    label = 'misinformation' if pred == 0 else 'true'

    return jsonify({'prediction': str(pred), 'label': label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
