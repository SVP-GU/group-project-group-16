import pandas as pd
import os

# Läs in data
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, 'crawled_articles.csv')
df = pd.read_csv(csv_path)

# Visa fördelning av etiketter
print("\nFördelning av etiketter:")
print(df['label'].value_counts())
print("\nTotal antal artiklar:", len(df))

# Visa några exempel på artiklar
print("\nExempel på sann artikel (label=1):")
true_article = df[df['label'] == 1].iloc[0]
print(f"Title: {true_article['title']}")
print(f"Text: {true_article['text'][:200]}...")

print("\nExempel på missinformation artikel (label=0):")
false_article = df[df['label'] == 0].iloc[0]
print(f"Title: {false_article['title']}")
print(f"Text: {false_article['text'][:200]}...")

# Kontrollera för tomma eller NULL-värden
print("\nAntal tomma eller NULL-värden per kolumn:")
print(df.isnull().sum())

# Kontrollera textlängder
df['text_length'] = df['text'].str.len()
print("\nStatistik över textlängder:")
print(df.groupby('label')['text_length'].describe())

# Få absolut sökväg till euvsdisinfo_base.csv
file_path = r"C:\Users\mirac\group-project-group-16\euvsdisinfo_base.csv"

# Läs in EUvsDisinfo datan
print("Läser in EUvsDisinfo data...")
df = pd.read_csv(file_path)

# Visa information om datasetet
print("\nDataset information:")
print(f"Antal rader: {len(df)}")
print(f"\nKolumner:")
print(df.columns.tolist())

print("\nFörsta raden:")
print(df.iloc[0]) 