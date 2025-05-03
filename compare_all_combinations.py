import joblib

# Ladda logreg-modellen
logreg_model = joblib.load('model/logreg_model.pkl')

# Ladda vectorizers
vectorizer1 = joblib.load('model/tfidf_vectorizer1.pkl')
vectorizer2 = joblib.load('model/tfidf_vectorizer2.pkl')

# Testmeningar
test_texts = [
    'The Earth revolves around the Sun.',
    'Water boils at 100 degrees Celsius.',
    'Barack Obama was the 44th President of the United States.',
    'Sweden is a country in Northern Europe.',
    'Vaccines cause autism.',
    'The moon landing was faked by NASA.',
    'COVID-19 was created in a lab as a bioweapon.',
    'Climate change is a hoax invented by the Chinese.',
    'Drinking bleach can cure coronavirus.',
    '5G spreads COVID-19.',
    'Politicians lie all the time.',
    "Mainstream media can't be trusted.",
    'Aliens are controlling world governments.'
]

# Testa logreg med båda vectorizers
for label, vectorizer in [('Vectorizer1', vectorizer1), ('Vectorizer2', vectorizer2)]:
    print('=' * 80)
    print(f'Testing: LogReg + {label}')
    print('=' * 80)
    for text in test_texts:
        try:
            X = vectorizer.transform([text])
            pred = logreg_model.predict(X)[0]
            print(f"Text: {text}")
            print(f"Prediction: {pred} ({'misinformation' if pred == 1 else 'true'})")
            print('-' * 60)
        except Exception as e:
            print(f"Error testing '{text}' → {e}")
            print('-' * 60)
