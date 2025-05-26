import pandas as pd

# Läs in CSV-filen
df = pd.read_csv('/Users/admin/Desktop/Grupparbete/GitHubRepo/group-project-group-16/grupp_sista (kopia)/crawled_articles_20250523_033432.csv')

# Visa ursprunglig storlek
print("Ursprunglig storlek:")
print(df.shape)

# Ta bort onödiga kolumner
columns_to_keep = ['text', 'label', 'source']
df = df[columns_to_keep]

# Visa storlek efter borttagning av kolumner
print("\nStorlek efter borttagning av kolumner:")
print(df.shape)

# Ta bort dubletter
df_no_duplicates = df.drop_duplicates(subset=['text'])

# Visa storlek efter borttagning av dubletter
print("\nStorlek efter borttagning av dubletter:")
print(df_no_duplicates.shape)

# Visa antal borttagna dubletter
print("\nAntal borttagna dubletter:")
print(df.shape[0] - df_no_duplicates.shape[0])

# Visa fördelning av labels efter borttagning av dubletter
print("\nFördelning av labels efter borttagning av dubletter:")
print(df_no_duplicates['label'].value_counts())
print("\nProcentuell fördelning av labels efter borttagning av dubletter:")
print(df_no_duplicates['label'].value_counts(normalize=True) * 100)

# Kontrollera saknade värden
print("\nSaknade värden per kolumn:")
print(df_no_duplicates.isnull().sum())

# Analysera källor (sources)
print("\nAntal unika källor:")
print(df_no_duplicates['source'].nunique())
print("\nTop 10 källor och antal artiklar:")
print(df_no_duplicates['source'].value_counts().head(10))

# Analysera textlängd
df_no_duplicates['text_length'] = df_no_duplicates['text'].str.len()
print("\nStatistik för textlängd:")
print(df_no_duplicates['text_length'].describe())

# Analysera fördelning av labels per källa
print("\nFördelning av labels per källa (top 5 källor):")
top_sources = df_no_duplicates['source'].value_counts().head(5).index
for source in top_sources:
    source_data = df_no_duplicates[df_no_duplicates['source'] == source]
    print(f"\n{source}:")
    print(source_data['label'].value_counts(normalize=True) * 100)

# Ta bort text_length kolumnen innan vi sparar
df_no_duplicates = df_no_duplicates.drop('text_length', axis=1)

# Spara den rensade datan till en ny CSV-fil
df_no_duplicates.to_csv('cleaned_articles.csv', index=False)
print("\nRensad data har sparats till 'cleaned_articles.csv'")