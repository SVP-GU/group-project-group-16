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
    model_name = "model/bert_model"
    tokenizer = AutoTokenizer.from_pretrained('model/bert_model')
    model = AutoModelForSequenceClassification.from_pretrained('model/bert_model')
    return model, tokenizer

def predict(text, model, tokenizer, threshold=0.75):
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
        st.error(f"Kunde inte ladda modellen. Fel: {str(e)}")
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
                st.success(f"Denna artikel bör vara sann")
            else:
                st.error(f"Denna artikel kan innehålla inslag av misinformation")
        else:
            st.warning("Vänligen klistra in en artikeltext först.")

if __name__ == "__main__":
    main() 