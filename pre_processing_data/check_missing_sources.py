import pandas as pd
import numpy as np

def main():
    print("🔍 Checking for missing sources...")
    
    # Läs datasetet
    print("\n1️⃣ Loading dataset...")
    df = pd.read_csv("data/combined_dataset_no_duplicates.csv")
    print(f"Dataset shape: {df.shape}")
    
    # Kontrollera saknade källor
    print("\n2️⃣ Checking for missing sources...")
    missing_sources = df[df['source'].isna()]
    print(f"Number of data points without source: {len(missing_sources)}")
    
    if len(missing_sources) > 0:
        print("\nLabel distribution for data points without source:")
        missing_source_labels = missing_sources['label'].value_counts()
        print(missing_source_labels)
        print("\nPercentages:")
        print((missing_source_labels / len(missing_sources) * 100).round(2))
        
        print("\nSample of data points without source:")
        print(missing_sources[['text', 'label']].head())
    
    # Visa fördelning av källor för hela datasetet
    print("\n3️⃣ Source distribution in entire dataset:")
    source_counts = df['source'].value_counts()
    print(source_counts)
    print("\nPercentages:")
    print((source_counts / len(df) * 100).round(2))
    
    # Visa fördelning av etiketter per källa
    print("\n4️⃣ Label distribution per source:")
    for source in df['source'].unique():
        if pd.isna(source):
            continue
        source_data = df[df['source'] == source]
        print(f"\nSource: {source}")
        label_counts = source_data['label'].value_counts()
        print(label_counts)
        print("Percentages:")
        print((label_counts / len(source_data) * 100).round(2))

if __name__ == "__main__":
    main() 