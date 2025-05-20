import pandas as pd
import re
from transformers import DistilBertTokenizer

def clean_text(text):
    # Ta bort specialtecken och extra mellanslag
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def main():
    print("🔄 Preprocessing text data...")
    
    # Läs datasetet
    df = pd.read_csv("data/combined_dataset_final.csv")
    print(f"Original dataset shape: {df.shape}")
    
    # Rensa texterna
    print("\n1️⃣ Cleaning texts...")
    df['text'] = df['text'].apply(clean_text)
    
    # Trunkera till max 512 tokens
    print("\n2️⃣ Truncating texts to max 512 tokens...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    def truncate_text(text):
        tokens = tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True)
        return tokenizer.decode(tokens, skip_special_tokens=True)
    
    df['text'] = df['text'].apply(truncate_text)
    
    # Ta bort korta texter (mindre än 10 tokens)
    print("\n3️⃣ Removing short texts...")
    df['token_count'] = df['text'].apply(lambda x: len(tokenizer.encode(x, add_special_tokens=True)))
    df = df[df['token_count'] >= 10]
    df = df.drop('token_count', axis=1)
    
    print(f"Dataset shape after preprocessing: {df.shape}")
    print(f"Removed {len(df) - df.shape[0]} short texts.")
    
    # Visa fördelning av etiketter och källor
    print("\nLabel distribution:")
    print(df['label'].value_counts())
    print("\nSource distribution:")
    print(df['source'].value_counts())
    
    # Spara det förbehandlade datasetet
    output_file = "data/combined_dataset_preprocessed.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Preprocessed dataset saved to: {output_file}")

if __name__ == "__main__":
    main() 