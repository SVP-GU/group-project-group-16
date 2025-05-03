from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

app = Flask(__name__)
CORS(app)

device = torch.device('cpu')

model_path = 'model/bert_model'
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    data = request.get_json()
    texts = data.get('texts', [])

    if not texts:
        return jsonify({'error': 'Ingen textlista skickades'}), 400

    flagged_indices = []
    for i, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            if pred == 0:  # 0 = misinformation
                flagged_indices.append(i)

    return jsonify({'flagged': flagged_indices})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
