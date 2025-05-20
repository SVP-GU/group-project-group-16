import pandas as pd
import numpy as np

def main():
    print("📊 Analysis of reprocessed dataset")
    print("=" * 50)
    
    # Läs datasetet
    print("\n1️⃣ Loading dataset...")
    df = pd.read_csv("data/combined_dataset_reprocessed.csv")
    print(f"Total number of samples: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Kontrollera saknade källor
    print("\n2️⃣ Checking for missing sources...")
    missing_sources = df[df['source'].isna()]
    print(f"Number of data points without source: {len(missing_sources)}")
    if len(missing_sources) > 0:
        print("\nSample of data points without source:")
        print(missing_sources[['text', 'label']].head())
    
    # Fördelning av etiketter
    print("\n3️⃣ Label distribution:")
    label_counts = df['label'].value_counts()
    print(label_counts)
    print("Percentages:")
    print((label_counts / len(df) * 100).round(2))
    
    # Fördelning av källor
    print("\n4️⃣ Source distribution:")
    source_counts = df['source'].value_counts()
    print(source_counts)
    print("Percentages:")
    print((source_counts / len(df) * 100).round(2))
    
    # Fördelning av etiketter per källa
    print("\n5️⃣ Label distribution per source:")
    for source in df['source'].unique():
        if pd.isna(source):
            continue
        source_data = df[df['source'] == source]
        print(f"\nSource: {source}")
        label_counts = source_data['label'].value_counts()
        print(label_counts)
        print("Percentages:")
        print((label_counts / len(source_data) * 100).round(2))
    
    # Exempel på texter
    print("\n6️⃣ Sample texts:")
    print(df[['text', 'label', 'source']].sample(5, random_state=42))

if __name__ == "__main__":
    main() 