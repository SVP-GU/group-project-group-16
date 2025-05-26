import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Konfigurera sidan
st.set_page_config(
    page_title="Nyhetsartikel Klassificerare",
    page_icon="📰",
    layout="centered"
)

# Ladda modell och tokenizer
@st.cache_resource
def load_model():
    model_name = "Mirac1999/swedish-news-classifier"
    tokenizer = AutoTokenizer.from_pretrained("Mirac1999/swedish-news-classifier")
    model = AutoModelForSequenceClassification.from_pretrained("Mirac1999/swedish-news-classifier")
    return model, tokenizer

def predict(text, model, tokenizer):
    """
    Gör en prediktion på given text.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return predictions[0].numpy()

# Huvudapplikation
def main():
    st.title("📰 Nyhetsartikel Klassificerare")
    st.write("Klistra in en nyhetsartikel för att analysera dess trovärdighet.")
    
    # Ladda modell
    try:
        model, tokenizer = load_model()
    except Exception as e:
        st.error("Kunde inte ladda modellen. Kontrollera att modellen är tränad och sparad i 'saved_model' mappen.")
        return
    
    # Textinmatning
    article_text = st.text_area("Klistra in artikeltext här:", height=200)
    
    if st.button("Analysera"):
        if article_text:
            # Gör prediktion
            predictions = predict(article_text, model, tokenizer)
            trovardig_score = predictions[1]  # Index 1 är för trovärdig
            
            # Visa resultat
            st.write("---")
            st.subheader("Resultat")
            
            if trovardig_score > 0.5:
                st.success(f"Denna artikel har inte flaggats, men tänk på att misinformation även kan spridas subtilt")
            else:
                st.error(f"Denna artikel har flaggats och kan innehålla misinformation.")
        else:
            st.warning("Vänligen klistra in en artikeltext först.")

if __name__ == "__main__":
    main() 

# -----------------------------
# Källanalys – domänkontroll
# -----------------------------
from urllib.parse import urlparse

MISINFO_DOMAINS = [
    "swebbtv.se",
    "friatider.se",
    "nyheteridag.se",
    "samnytt.se",
    "newsvoice.se",
    "nyatider.nu",
    "exakt24.se",
    "dissidenter.se",
    "projektsanning.com",
    "mindsverige.se"
]

TRUSTED_DOMAINS = [
    "dn.se", "svd.se", "reuters.com", "bbc.com", "nytimes.com", "ur.se"
]

def extract_domain(url):
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return None

st.subheader("Källanalys – hur trovärdig är länken?")
st.markdown("🔎 **Tänk på att skriva in din länk i rätt format, t.ex.: `https://www.hemsidan.se`**")
url_input = st.text_input("Klistra in en nyhetslänk:")

if url_input:
    domain = extract_domain(url_input)
    if domain in MISINFO_DOMAINS:
        st.error(f"⚠️ Varning: {domain} är känd för att sprida misinformation.")
        st.markdown("""
                *Domänerna som flaggas är sådana som återkommande förekommer i faktagranskningar och forskningsrapporter från t.ex. **MSB** och **FOI**.  
                Det betyder inte att allt innehåll på dessa sidor är falskt, utan att de ofta är källor till missvisande eller felaktig information.*
                """)
        st.markdown("**Prova istället att läsa från:**")
        for trusted in TRUSTED_DOMAINS:
            st.markdown(f"- [https://{trusted}](https://{trusted})")
    elif domain in TRUSTED_DOMAINS:
        st.success(f"✅ {domain} är en etablerad och pålitlig källa.")
    else:
        st.info(f"ℹ️ {domain} finns inte i vår databas – ingen känd flaggning.")

# Ändrar bakgrundsfärg
st.markdown("""
    <style>
    .stApp {
        background-color: #0f172a;
        background-image: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3 {
        color: #38bdf8;
    }
    .reportview-container .markdown-text-container {
        font-size: 18px;
    }
    .stTextInput>div>div>input {
        background-color: #1e293b;
        color: #e2e8f0;
    }
    .stButton>button {
        background-color: #38bdf8;
        color: #0f172a;
    }
    </style>
""", unsafe_allow_html=True)