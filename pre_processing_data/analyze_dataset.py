import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_text_lengths(texts):
    lengths = [len(str(text).split()) for text in texts]
    return {
        'min': min(lengths),
        'max': max(lengths),
        'mean': np.mean(lengths),
        'median': np.median(lengths),
        'std': np.std(lengths)
    }

def main():
    print("📊 Dataset Analysis Report")
    print("=" * 50)
    
    # Läs datasetet
    print("\n1️⃣ Loading dataset...")
    df = pd.read_csv("data/combined_dataset_extended.csv")
    
    # Grundläggande statistik
    print("\n2️⃣ Basic Statistics:")
    print(f"Total number of samples: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nColumns in dataset:")
    for col in df.columns:
        print(f"- {col}")
    
    # Analysera etiketter
    print("\n3️⃣ Label Distribution:")
    label_counts = df['label'].value_counts()
    print("\nLabel counts:")
    print(label_counts)
    print("\nLabel percentages:")
    print((label_counts / len(df) * 100).round(2))
    
    # Analysera textlängder
    print("\n4️⃣ Text Length Statistics (in words):")
    length_stats = analyze_text_lengths(df['text'])
    for stat, value in length_stats.items():
        print(f"{stat.capitalize()}: {value:.2f}")
    
    # Analysera källor om de finns
    if 'source' in df.columns:
        print("\n5️⃣ Source Distribution:")
        source_counts = df['source'].value_counts()
        print("\nSamples per source:")
        print(source_counts)
        print("\nSource percentages:")
        print((source_counts / len(df) * 100).round(2))
    
    # Kontrollera för saknade värden
    print("\n6️⃣ Missing Values Check:")
    missing_values = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing_values[missing_values > 0])
    
    # Skapa visualiseringar
    print("\n7️⃣ Creating visualizations...")
    
    # Skapa en mapp för visualiseringar om den inte finns
    import os
    if not os.path.exists('analysis_plots'):
        os.makedirs('analysis_plots')
    
    # Label distribution plot
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='label')
    plt.title('Distribution of Labels')
    plt.savefig('analysis_plots/label_distribution.png')
    plt.close()
    
    # Text length distribution plot
    plt.figure(figsize=(10, 6))
    text_lengths = [len(str(text).split()) for text in df['text']]
    sns.histplot(text_lengths, bins=50)
    plt.title('Distribution of Text Lengths')
    plt.xlabel('Number of Words')
    plt.savefig('analysis_plots/text_length_distribution.png')
    plt.close()
    
    if 'source' in df.columns:
        # Source distribution plot
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, y='source')
        plt.title('Distribution of Sources')
        plt.savefig('analysis_plots/source_distribution.png')
        plt.close()
    
    print("\n✅ Analysis complete! Visualizations saved in 'analysis_plots' directory.")

if __name__ == "__main__":
    main() 