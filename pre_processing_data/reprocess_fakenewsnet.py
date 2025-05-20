import pandas as pd
import numpy as np

def main():
    print("🔄 Reprocessing FakeNewsNet dataset...")
    
    # Läs det nuvarande datasetet
    print("\n1️⃣ Loading current dataset...")
    df = pd.read_csv("data/combined_dataset_no_duplicates.csv")
    print(f"Current dataset shape: {df.shape}")
    
    # Ta bort FakeNewsNet data
    print("\n2️⃣ Removing current FakeNewsNet data...")
    df_without_fakenewsnet = df[df['source'] != 'FakeNewsNet']
    print(f"Dataset shape after removing FakeNewsNet: {df_without_fakenewsnet.shape}")
    
    # Läs in FakeNewsNet data på nytt
    print("\n3️⃣ Loading fresh FakeNewsNet data...")
    fakenews_files = [
        ('data/fakenewsnet/gossipcop_fake.csv', 1),
        ('data/fakenewsnet/gossipcop_real.csv', 0),
        ('data/fakenewsnet/politifact_fake.csv', 1),
        ('data/fakenewsnet/politifact_real.csv', 0)
    ]
    
    fakenews_list = []
    for file, label in fakenews_files:
        print(f"\nProcessing {file}...")
        try:
            df_file = pd.read_csv(file)
            print(f"Columns in file: {df_file.columns.tolist()}")
            print(f"Number of rows: {len(df_file)}")
            
            # Skapa textfältet (title + text)
            if 'title' in df_file.columns and 'text' in df_file.columns:
                df_file['text'] = df_file['title'].fillna('') + ' ' + df_file['text'].fillna('')
            elif 'title' in df_file.columns:
                df_file['text'] = df_file['title']
            elif 'text' in df_file.columns:
                df_file['text'] = df_file['text']
            else:
                print(f"Warning: No text column found in {file}")
                continue
            
            # Sätt etikett
            df_file['label'] = label
            df_file['source'] = 'FakeNewsNet'
            
            # Visa exempel på data
            print("\nSample of data:")
            print(df_file[['text', 'label']].head())
            
            fakenews_list.append(df_file[['text', 'label', 'source']])
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue
    
    # Kombinera alla FakeNewsNet data
    fakenews_df = pd.concat(fakenews_list, ignore_index=True)
    print(f"\nFakeNewsNet data shape: {fakenews_df.shape}")
    
    # Visa fördelning av etiketter i FakeNewsNet
    print("\nFakeNewsNet label distribution:")
    fakenews_labels = fakenews_df['label'].value_counts()
    print(fakenews_labels)
    print("\nPercentages:")
    print((fakenews_labels / len(fakenews_df) * 100).round(2))
    
    # Kombinera med huvuddatasetet
    print("\n4️⃣ Combining datasets...")
    final_df = pd.concat([df_without_fakenewsnet, fakenews_df], ignore_index=True)
    
    # Visa slutlig fördelning
    print("\nFinal dataset statistics:")
    print(f"Total shape: {final_df.shape}")
    
    print("\nLabel distribution:")
    final_labels = final_df['label'].value_counts()
    print(final_labels)
    print("\nPercentages:")
    print((final_labels / len(final_df) * 100).round(2))
    
    print("\nSource distribution:")
    final_sources = final_df['source'].value_counts()
    print(final_sources)
    print("\nPercentages:")
    print((final_sources / len(final_df) * 100).round(2))
    
    # Spara det nya datasetet
    output_file = "data/combined_dataset_reprocessed.csv"
    final_df.to_csv(output_file, index=False)
    print(f"\n✅ New dataset saved to: {output_file}")

if __name__ == "__main__":
    main() 