from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Ladda modell och vectorizer (justera namnen om du vill använda andra)
model = joblib.load('model/logreg_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer1.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'Ingen text skickades'}), 400

    # Transformera text
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]

    return jsonify({'prediction': str(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
