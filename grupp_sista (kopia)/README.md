# Nyhetsartikel Klassificerare

Detta projekt är en AI-baserad lösning för att klassificera nyhetsartiklar som antingen trovärdiga eller misinformation. Projektet består av tre huvudkomponenter:

1. Webbskrapning av nyhetsartiklar
2. Träning av en DistilBERT-modell
3. Streamlit-app för användarinteraktion

## Installation

1. Klona projektet:
```bash
git clone [repository-url]
cd [project-directory]
```

2. Installera beroenden:
```bash
pip install -r requirements.txt
```

## Användning

1. Kör webbskrapningen:
```bash
python crawler.py
```

2. Träna modellen:
```bash
python model.py
```

3. Starta Streamlit-appen:
```bash
streamlit run app.py
```

## Projektstruktur

- `crawler.py`: Webbskrapningslogik för att samla in nyhetsartiklar
- `model.py`: Träning och validering av DistilBERT-modellen
- `app.py`: Streamlit-applikation för användarinteraktion
- `requirements.txt`: Projektberoenden
- `crawled_articles.csv`: Dataset med skrapade artiklar
- `saved_model/`: Mapp för sparade modeller och tokenizer

## Tekniska Detaljer

- Använder DistilBERT för textklassificering
- Tränar på svenska och engelska nyhetsartiklar
- Implementerar 80/20 train/test split
- Använder Streamlit för användargränssnitt

## Framtida Förbättringar

- Implementering av robust felhantering
- Språkhantering för svenska/engelska
- Modellvalidering med metrics
- Caching i Streamlit-appen
- Confidence scores i prediktioner 