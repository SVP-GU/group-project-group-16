from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Ladda modellen och tokenizer
model_path = "trained_models/bert_model_20250525_001044"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

@app.route('/classify', methods=['POST'])
def classify_text():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        # Tokenize och förbered input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        
        # Gör förutsägelse
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = predictions[0].numpy()
            
        # Konvertera till procent
        confidence = float(prediction[1]) * 100
        
        return jsonify({
            'is_fake': bool(prediction[1] > prediction[0]),
            'confidence': confidence,
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 