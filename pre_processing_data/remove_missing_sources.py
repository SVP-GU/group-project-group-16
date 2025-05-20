import pandas as pd

def main():
    print("🔄 Removing data points without source...")
    
    # Läs datasetet
    df = pd.read_csv("data/combined_dataset_reprocessed.csv")
    print(f"Original dataset shape: {df.shape}")
    
    # Ta bort datapunkter utan källa
    df_clean = df.dropna(subset=['source'])
    print(f"Dataset shape after removing missing sources: {df_clean.shape}")
    print(f"Removed {len(df) - len(df_clean)} data points without source.")
    
    # Visa fördelning av etiketter och källor
    print("\nLabel distribution:")
    print(df_clean['label'].value_counts())
    print("\nSource distribution:")
    print(df_clean['source'].value_counts())
    
    # Spara det rensade datasetet
    output_file = "data/combined_dataset_final.csv"
    df_clean.to_csv(output_file, index=False)
    print(f"\n✅ Cleaned dataset saved to: {output_file}")

if __name__ == "__main__":
    main() 