import pandas as pd

# Ladda ditt gamla kombinerade dataset
combined = pd.read_csv('data/combined_dataset.csv')
print(f"✅ Existerande dataset: {combined.shape[0]} rader")

# Ladda nya fake_or_real_news-datasetet
new_data = pd.read_csv('data/fake_or_real_news.csv')
print(f"✅ Nya datasetet: {new_data.shape[0]} rader")

# Skapa textfältet (title + text)
new_data['text'] = new_data['title'].fillna('') + ' ' + new_data['text'].fillna('')

# Mappa labels
label_map = {'FAKE': 0, 'REAL': 1}
new_data['label'] = new_data['label'].map(label_map)

# Behåll bara relevanta kolumner
new_data = new_data[['text', 'label']]

# Slå ihop
final_combined = pd.concat([combined, new_data], ignore_index=True)
print(f"✅ Totalt kombinerat dataset: {final_combined.shape[0]} rader")

# Spara som ny fil
final_combined.to_csv('data/combined_dataset_extended.csv', index=False)
print("✅ Ny kombinerad fil sparad: data/combined_dataset_extended.csv")
