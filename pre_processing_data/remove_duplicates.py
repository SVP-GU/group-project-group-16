import pandas as pd
import numpy as np

def main():
    print("🔄 Starting duplicate removal process...")
    
    # Läs datasetet
    print("\n1️⃣ Loading dataset...")
    df = pd.read_csv("data/combined_dataset_preprocessed.csv")
    print(f"Original dataset shape: {df.shape}")
    
    # Kontrollera dubletter baserat på text
    print("\n2️⃣ Checking for duplicates...")
    duplicates = df[df.duplicated(subset=['text'], keep=False)]
    print(f"Number of duplicate texts: {len(duplicates)}")
    
    if len(duplicates) > 0:
        print("\nSample of duplicate texts:")
        # Gruppera efter text och visa exempel
        duplicate_groups = duplicates.groupby('text').agg({
            'label': lambda x: list(x),
            'source': lambda x: list(x) if 'source' in df.columns else None
        }).head(5)
        print(duplicate_groups)
        
        # Analysera etiketter för dubletter
        print("\nLabel distribution in duplicates:")
        duplicate_label_counts = duplicates['label'].value_counts()
        print(duplicate_label_counts)
        print("\nPercentages in duplicates:")
        print((duplicate_label_counts / len(duplicates) * 100).round(2))
    
    # Ta bort dubletter
    print("\n3️⃣ Removing duplicates...")
    df_no_duplicates = df.drop_duplicates(subset=['text'], keep='first')
    
    # Visa förändringar
    print(f"\nRemoved {len(df) - len(df_no_duplicates)} duplicate texts")
    print(f"Final dataset shape: {df_no_duplicates.shape}")
    
    # Visa fördelning av etiketter efter borttagning av dubletter
    print("\nLabel distribution after removing duplicates:")
    label_counts = df_no_duplicates['label'].value_counts()
    print(label_counts)
    print("\nPercentages:")
    print((label_counts / len(df_no_duplicates) * 100).round(2))
    
    # Spara det rensade datasetet
    output_file = "data/combined_dataset_no_duplicates.csv"
    df_no_duplicates.to_csv(output_file, index=False)
    print(f"\n✅ Cleaned dataset saved to: {output_file}")

if __name__ == "__main__":
    main() 