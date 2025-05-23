import pandas as pd
from googletrans import Translator
import time
import logging
from tqdm import tqdm
import os

# Sätt upp logging
logging.basicConfig(
    filename=f'combine_datasets_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def translate_to_swedish(text, translator):
    """
    Översätter text från engelska till svenska
    """
    try:
        if not isinstance(text, str) or not text.strip():
            return ''
        
        # Dela upp texten i mindre bitar för att undvika översättningsgränser
        max_length = 5000
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        
        translated_chunks = []
        for chunk in chunks:
            try:
                translated = translator.translate(chunk, dest='sv', src='en')
                translated_chunks.append(translated.text)
                # Lägg till en kort paus mellan översättningar
                time.sleep(1)
            except Exception as e:
                logging.error(f"Fel vid översättning av chunk: {str(e)}")
                translated_chunks.append(chunk)
        
        return ' '.join(translated_chunks)
    except Exception as e:
        logging.error(f"Fel vid översättning: {str(e)}")
        return text

def main():
    # Initiera översättare
    translator = Translator()
    
    # Läs in den crawlade svenska datan
    script_dir = os.path.dirname(os.path.abspath(__file__))
    crawled_files = [f for f in os.listdir(script_dir) if f.startswith('crawled_articles_') and f.endswith('.csv')]
    
    if not crawled_files:
        print("Ingen crawlad data hittades!")
        return
        
    latest_crawled = max(crawled_files)
    swedish_df = pd.read_csv(os.path.join(script_dir, latest_crawled))
    print(f"Läste in {len(swedish_df)} svenska artiklar")
    
    # Läs in EUvsDisinfo datan
    eu_df = pd.read_csv('../euvsdisinfo_base.csv')
    print(f"Läste in {len(eu_df)} artiklar från EUvsDisinfo")
    
    # Välj relevanta kolumner och översätt innehållet
    print("\nÖversätter EUvsDisinfo artiklar till svenska...")
    eu_df['title_sv'] = pd.Series(dtype=str)
    eu_df['text_sv'] = pd.Series(dtype=str)
    
    for idx, row in tqdm(eu_df.iterrows(), total=len(eu_df)):
        eu_df.at[idx, 'title_sv'] = translate_to_swedish(row['title'], translator)
        eu_df.at[idx, 'text_sv'] = translate_to_swedish(row['text'], translator)
    
    # Formatera EUvsDisinfo data för att matcha den svenska datan
    eu_formatted = pd.DataFrame({
        'url': eu_df['url'],
        'title': eu_df['title_sv'],
        'text': eu_df['text_sv'],
        'label': 0,  # Alla EUvsDisinfo artiklar är desinformation
        'source': 'euvsdisinfo',
        'translated': True
    })
    
    # Kombinera dataseten
    combined_df = pd.concat([swedish_df, eu_formatted], ignore_index=True)
    
    # Spara det kombinerade datasetet
    output_file = os.path.join(script_dir, 'combined_dataset.csv')
    combined_df.to_csv(output_file, index=False)
    
    print(f"\nStatistik för kombinerat dataset:")
    print(f"Totalt antal artiklar: {len(combined_df)}")
    print(f"Antal trovärdiga artiklar: {len(combined_df[combined_df['label'] == 1])}")
    print(f"Antal desinformationsartiklar: {len(combined_df[combined_df['label'] == 0])}")
    print(f"\nDatasetet har sparats som: {output_file}")

if __name__ == "__main__":
    main() 