import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# Layout
st.set_page_config(page_title="Källkritikradarn", layout="wide")

# Custom CSS for design
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background-color: #fdfbff;
        color: #311b92;
    }
    .stApp {
        padding: 2rem;
    }
    h1, h2, h3, h4 {
        color: #512da8;
    }
    .metric-label {
        font-size: 14px;
        color: #4527a0;
    }
    </style>
""", unsafe_allow_html=True)

# SIDHUVUD
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Quicksand:wght@600&display=swap" rel="stylesheet">

<h1 style='
    font-family: "Quicksand", sans-serif;
    font-size: 48px;
    font-weight: 600;
    background: linear-gradient(to right, #ba68c8, #9575cd, #7986cb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2em;
'>
Källkritikradarn
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<div style='font-size:20px; font-family:Quicksand, sans-serif; color:#6a1b9a;'>
    <strong>Visualisering av statistik och analys som visar behovet av ett digitalt verktyg för att identifiera misinformation bland ungdomar.</strong>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# RAD 1 – Cirkeldiagram och Stapeldiagram
col1, col2 = st.columns([1, 1])  # Jämn fördelning, tillräckligt plats med mindre figurer

with col1:
    st.markdown("""
<h3 style='color:#6a1b9a; font-family:Quicksand, sans-serif; font-weight:600;'>
Hur många kontrollerar information? (SCB)
</h3>
""", unsafe_allow_html=True)
    labels = ['Kontrollerar', 'Kontrollerar inte']
    sizes = [38, 62]
    colors = ['#b388ff', '#e1bee7']

    def make_autopct(values):
        def my_autopct(pct):
            return f"{int(round(pct))}%"
        return my_autopct

    fig1, ax1 = plt.subplots(figsize=(5.5, 4))
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, autopct=make_autopct(sizes),
        colors=colors, startangle=90, textprops={'fontsize': 12, 'color': 'white'}
    )
    ax1.axis('equal')
    st.pyplot(fig1)
    st.caption("Endast 38% kontrollerar tveksam information enligt SCB (2023).")

with col2:
    st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Quicksand:wght@600&display=swap" rel="stylesheet">
<h3 style='
    color:#6a1b9a;
    font-family:"Quicksand", sans-serif;
    font-weight:600;
    margin-bottom:0.5em;
'>
Andel personer som sett missvisande information (SCB, 2023)
</h3>
""", unsafe_allow_html=True)

    data = {
        'Kategori': [
            "16–24 år", 
            "25–34 år", 
            "35–44 år", 
            "45–54 år", 
            "55–64 år", 
            "65–74 år", 
            "75–85 år"
        ],
        'Procent': [65, 77, 68, 60, 55, 50, 38]
    }

    df = pd.DataFrame(data)
    colors = plt.cm.magma(np.linspace(0.3, 0.7, len(df)))  # Gradientfärg

    fig2, ax2 = plt.subplots(figsize=(5.5, 4))
    bars = ax2.barh(df['Kategori'], df['Procent'], color=colors)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Procent", fontsize=12)
    ax2.set_title("Missvisande information online – per åldersgrupp", fontsize=14, color='#6a1b9a')

    for bar, percent in zip(bars, df['Procent']):
        ax2.text(percent + 1, bar.get_y() + bar.get_height() / 2, f"{percent}%", va='center', fontsize=10, color="#4a148c")

    ax2.tick_params(colors='#4a148c')
    for spine in ax2.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    st.pyplot(fig2)
    st.caption("SCB (2023): Yngre personer är mest exponerade för missvisande information på internet.")



# --- RAD 2: Visualisering av nyhetsfördelning och modellprestanda ---
st.markdown("---")
st.markdown("""
<h3 style='color:#6a1b9a; font-family:Quicksand, sans-serif; font-weight:600;'>
Analys av dataset och modellprestanda
</h3>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1.2, 1])

with col1:
    labels = ['Sanna nyheter', 'Falska/vinklade nyheter']
    values = [25247, 16542]
    colors = ['#dfc4f5', '#ba68c8']

    fig4, ax4 = plt.subplots(figsize=(6, 4))
    bars = ax4.bar(labels, values, color=colors, edgecolor='black', linewidth=0.8)

    ax4.set_ylabel("Antal artiklar", fontsize=13, color="#4a148c")
    ax4.set_title("Fördelning av nyhetstyper i träningsdata", fontsize=14, color="#6a1b9a")
    ax4.tick_params(colors='#4a148c')

    for bar, val in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width() / 2, val + 100, str(val),
                 ha='center', fontsize=12, color="#4a148c")

    ax4.set_facecolor('#fdfbff')
    for spine in ['top', 'right']:
        ax4.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax4.spines[spine].set_color('#b39ddb')

    plt.tight_layout()
    st.pyplot(fig4)

with col2:
    st.markdown("""
<div style='color:#6a1b9a; font-family:Quicksand, sans-serif; font-size:16px;'>
<strong>Klass 0 (Sanna nyheter)</strong><br>
• Precision: 0.83<br>
• Recall: 0.77<br>
• F1-score: 0.80<br>
• Support: 5050<br><br>

<strong>Klass 1 (Falska nyheter)</strong><br>
• Precision: 0.68<br>
• Recall: 0.75<br>
• F1-score: 0.72<br>
• Support: 3308<br><br>

<strong>Övergripande</strong><br>
• Accuracy: 0.76<br>
• Macro avg F1: 0.76<br>
• Weighted avg F1: 0.76<br>
• Validation loss: 0.49
</div>
""", unsafe_allow_html=True)




    with st.expander("Klicka för att läsa vår analys"):
        st.markdown("""
        - **Modellen presterar starkt** på att identifiera *sanna nyheter* (F1: 0.80).
        - **Falska nyheter** detekteras något svagare (F1: 0.72), men recall är hög (0.75) vilket tyder på att den hittar många av dem.
        - **Precision för falska nyheter är lägre (0.68)** vilket betyder att vissa felaktigt flaggas.
        - **Balans i metrik (macro/weighted F1 = 0.76)** visar jämn prestanda över båda klasser.
        - **Låg valideringsförlust (0.49)** indikerar att modellen generaliserar bra utan att överträna.
        """)


