import pandas as pd
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def extract_topics(text):
    # Konvertera till lowercase och tokenize
    tokens = word_tokenize(text.lower())
    
    # Ta bort stopwords och specialtecken
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    
    return tokens

def main():
    print("📊 Analyzing news categories in dataset...")
    
    # Ladda datasetet
    df = pd.read_csv("data/combined_dataset_preprocessed.csv")
    
    # Definiera nyhetskategorier och deras nyckelord
    categories = {
        'Politics': ['trump', 'biden', 'election', 'president', 'congress', 'senate', 'democrat', 'republican', 'campaign', 'vote'],
        'Economy': ['economy', 'market', 'stock', 'trade', 'business', 'financial', 'dollar', 'inflation', 'recession', 'bank'],
        'Health': ['health', 'medical', 'disease', 'virus', 'covid', 'vaccine', 'hospital', 'doctor', 'patient', 'treatment'],
        'Technology': ['technology', 'tech', 'digital', 'computer', 'software', 'internet', 'data', 'online', 'app', 'device'],
        'Entertainment': ['celebrity', 'movie', 'music', 'actor', 'actress', 'film', 'star', 'entertainment', 'show', 'concert'],
        'Sports': ['sport', 'game', 'team', 'player', 'coach', 'championship', 'league', 'match', 'tournament', 'athlete']
    }
    
    # Analysera varje kategori
    print("\nCategory Analysis:")
    print("=" * 50)
    
    for category, keywords in categories.items():
        # Räkna förekomster per källa
        category_counts = {source: 0 for source in df['source'].unique()}
        category_labels = {0: 0, 1: 0}  # 0 for true, 1 for fake
        
        for _, row in df.iterrows():
            text = row['text'].lower()
            if any(keyword in text for keyword in keywords):
                category_counts[row['source']] += 1
                category_labels[row['label']] += 1
        
        # Skriv ut resultat
        print(f"\n{category}:")
        print(f"Total occurrences: {sum(category_counts.values())}")
        print("Distribution by source:")
        for source, count in category_counts.items():
            if count > 0:
                print(f"- {source}: {count}")
        print("Distribution by label:")
        print(f"- True news (0): {category_labels[0]}")
        print(f"- Fake news (1): {category_labels[1]}")
        
        # Beräkna och visa procentuell fördelning
        total = sum(category_labels.values())
        if total > 0:
            print("Label percentages:")
            print(f"- True news: {(category_labels[0]/total)*100:.1f}%")
            print(f"- Fake news: {(category_labels[1]/total)*100:.1f}%")
    
    # Analysera vanligaste orden i varje källa
    print("\nMost common words by source:")
    print("=" * 50)
    
    for source in df['source'].unique():
        source_texts = df[df['source'] == source]['text']
        all_tokens = []
        for text in source_texts:
            all_tokens.extend(extract_topics(text))
        
        # Visa top 10 vanligaste orden
        word_freq = Counter(all_tokens).most_common(10)
        print(f"\n{source}:")
        for word, count in word_freq:
            print(f"- {word}: {count}")

if __name__ == "__main__":
    # Ladda nltk-resurser
    nltk.download('punkt')
    nltk.download('stopwords')
    main() 