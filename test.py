import requests

url = 'http://127.0.0.1:5000/predict'

test_sentences = [
    "The sky is blue.",
    "COVID-19 was created as a bioweapon.",
    "Water boils at 100 degrees Celsius.",
    "The moon landing was faked by NASA.",
    "Vaccines cause autism.",
    "Barack Obama was the 44th president of the USA.",
    "5G networks spread COVID-19.",
    "Climate change is real.",
    "Drinking bleach can cure coronavirus.",
    "Aliens control world governments."
]

for sentence in test_sentences:
    try:
        response = requests.post(url, json={'text': sentence}, timeout=10)
        response.raise_for_status()  # kastar fel om statuskod är inte 2xx
        result = response.json()
        print(f"Text: {sentence}")
        print(f"Prediction: {result['prediction']} → {result['label']}")
        print("-" * 50)
    except requests.exceptions.RequestException as e:
        print(f"Error for sentence: {sentence}")
        print(f"Exception: {e}")
        print("-" * 50)
