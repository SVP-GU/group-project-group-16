import pandas as pd
import numpy as np

def main():
    print("🔄 Starting label preprocessing...")
    
    # Läs datasetet
    print("\n1️⃣ Loading dataset...")
    df = pd.read_csv("data/combined_dataset_extended.csv")
    print(f"Original dataset shape: {df.shape}")
    
    # Kontrollera för saknade etiketter
    print("\n2️⃣ Checking for missing labels...")
    missing_labels = df[df['label'].isna()]
    print(f"Number of texts without labels: {len(missing_labels)}")
    
    if len(missing_labels) > 0:
        print("\nSample of texts without labels:")
        print(missing_labels[['text', 'label']].head())
    
    # Visa ursprunglig fördelning
    print("\n3️⃣ Original label distribution:")
    original_counts = df['label'].value_counts()
    print(original_counts)
    
    # Definiera mappning
    label_mapping = {
        # True/Trustworthy labels -> 0
        '0': 0,
        'true': 0,
        'trustworthy': 0,
        'mostly-true': 0,
        
        # False/Misinformation labels -> 1
        '1': 1,
        'disinformation': 1,
        'false': 1,
        'pants-fire': 1,
        'barely-true': 1,
        
        # Special case - half-true -> 1 (kan diskuteras)
        'half-true': 1
    }
    
    # Konvertera etiketter
    print("\n4️⃣ Converting labels...")
    df['label'] = df['label'].map(label_mapping)
    
    # Ta bort rader där etiketten inte kunde mappas (NaN)
    df = df.dropna(subset=['label'])
    
    # Konvertera label till int
    df['label'] = df['label'].astype(int)
    
    # Visa ny fördelning
    print("\n5️⃣ New label distribution:")
    new_counts = df['label'].value_counts()
    print(new_counts)
    print("\nPercentages:")
    print((new_counts / len(df) * 100).round(2))
    
    # Spara det förbehandlade datasetet
    output_file = "data/combined_dataset_binary_labels.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Preprocessed dataset saved to: {output_file}")
    print(f"Final dataset shape: {df.shape}")

if __name__ == "__main__":
    main() 