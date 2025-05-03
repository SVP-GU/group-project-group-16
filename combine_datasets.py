import pandas as pd

# EUvsDisinfo (CSV, ligger direkt i huvudmappen)
euvs = pd.read_csv('euvsdisinfo_base.csv')
euvs = euvs.rename(columns={'keywords': 'text', 'class': 'label'})
euvs['source'] = 'EUvsDisinfo'
euvs = euvs[['text', 'label', 'source']]

# LIAR (TSV: train, test, valid)
liar_files = ['data/liar_dataset/train.tsv', 'data/liar_dataset/test.tsv', 'data/liar_dataset/valid.tsv']
liar_list = []
for file in liar_files:
    df = pd.read_csv(file, sep='\t', header=None, names=['id', 'label', 'statement', 'subject', 'speaker', 'speaker_job', 'state', 'party', 'barely_true', 'false', 'half_true', 'mostly_true', 'pants_on_fire', 'context'])
    df = df.rename(columns={'statement': 'text'})
    liar_list.append(df[['text', 'label']])
liar = pd.concat(liar_list, ignore_index=True)
liar['source'] = 'LIAR'

# FakeNewsNet (CSV: gossipcop_fake, gossipcop_real, politifact_fake, politifact_real)
fakenews_files = [
    'data/fakenewsnet/gossipcop_fake.csv',
    'data/fakenewsnet/gossipcop_real.csv',
    'data/fakenewsnet/politifact_fake.csv',
    'data/fakenewsnet/politifact_real.csv'
]
fakenews_list = []
for file in fakenews_files:
    df = pd.read_csv(file)
    if 'title' in df.columns:
        df = df.rename(columns={'title': 'text'})
    else:
        raise ValueError(f"Filen {file} saknar 'title'-kolumn")
    # Assign label: fake = 0, real = 1 (based on filename)
    if 'fake' in file:
        df['label'] = 0
    else:
        df['label'] = 1
    fakenews_list.append(df[['text', 'label']])
fakenews = pd.concat(fakenews_list, ignore_index=True)
fakenews['source'] = 'FakeNewsNet'

# Kombinera allt
combined = pd.concat([euvs, liar, fakenews], ignore_index=True)
combined = combined.dropna(subset=['text', 'label']).reset_index(drop=True)

# Spara till CSV
combined.to_csv('data/combined_dataset.csv', index=False)

print(f"Klar! Totalt antal datapunkter: {len(combined)}")
print(combined['source'].value_counts())
