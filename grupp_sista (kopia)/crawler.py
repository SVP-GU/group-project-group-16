import pandas as pd
from newspaper import Article
from tqdm import tqdm
import time
import random

# Definiera källor och deras labels
SOURCES = {
    # Trovärdiga källor (label = 1)
    'https://www.dn.se': 1,
    'https://www.svd.se': 1,
    'https://www.gp.se': 1,
    # Misinfo källor (label = 0)
    'https://www.friatider.se': 0,
    'https://www.swebbtv.se': 0,
    'https://www.nyatider.nu/': 0,
    'https://www.samnytt.se/': 0
}

def extract_article(url):
    """
    Extraherar artikeldata från en given URL.
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        # Kontrollera om artikeln är tillräckligt lång
        if len(article.text) < 300:
            return None
            
        return {
            'url': url,
            'title': article.title,
            'text': article.text,
            'label': SOURCES[url.split('/')[2]]  # Extraherar domänen för label
        }
    except Exception as e:
        print(f"Fel vid extrahering av {url}: {str(e)}")
        return None

def crawl_articles():
    """
    Huvudfunktion för att skrapa artiklar från alla källor.
    """
    articles = []
    
    # Skapa en lista med alla källor
    urls = list(SOURCES.keys())
    
    # Skrapa artiklar med progress bar
    for url in tqdm(urls, desc="Skrapar artiklar"):
        article_data = extract_article(url)
        if article_data:
            articles.append(article_data)
        # Lägg till en liten paus för att undvika rate limiting
        time.sleep(random.uniform(1, 3))
    
    # Skapa DataFrame och spara till CSV
    if articles:
        df = pd.DataFrame(articles)
        df.to_csv('crawled_articles.csv', index=False)
        print(f"Skrapade {len(articles)} artiklar och sparade till crawled_articles.csv")
    else:
        print("Inga artiklar kunde skrapas.")

if __name__ == "__main__":
    crawl_articles() 