import requests

url = 'http://127.0.0.1:5000/predict'
data = {'text': 'detta är ett test'}

response = requests.post(url, json=data)

print('Svar från servern:', response.status_code)
print('Innehåll:', response.json())
