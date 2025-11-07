import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.boolean_retrieval import BooleanRetrieval
from src.vsm import VectorSpaceModel

st.title("Sistem Informasi Retrieval (IR) - UTS STKI")

# Pilih model
model_choice = st.selectbox("Pilih Model Retrieval:", ["Boolean Retrieval", "Vector Space Model"])

if model_choice == "Boolean Retrieval":
    model = BooleanRetrieval()
    query_type = "Boolean Query (e.g., 'demam AND batuk' or 'NOT alergi')"
else:
    model = VectorSpaceModel()
    query_type = "Query bebas (e.g., 'gejala demam')"

query = st.text_input(f"Masukkan {query_type}:")

if st.button("Cari"):
    if query:
        if model_choice == "Boolean Retrieval":
            results = model.search(query)
            if results:
                st.subheader("Hasil Pencarian:")
                for doc_id, snippet in results:
                    st.write(f"**{doc_id}**: {snippet}...")
            else:
                st.write("Tidak ada dokumen yang relevan.")
        else:
            results = model.search(query)
            if results:
                st.subheader("Hasil Pencarian:")
                for doc_id, snippet, score in results:
                    st.write(f"**{doc_id}** (Skor: {score:.2f}): {snippet}...")
            else:
                st.write("Tidak ada dokumen yang relevan.")
    else:
        st.write("Masukkan query terlebih dahulu.")

# Informasi tentang sistem
st.markdown("---")
st.subheader("Tentang Sistem")
st.write("""
Sistem Informasi Retrieval (IR) ini dibuat untuk UTS STKI menggunakan dataset teks tentang gejala penyakit dalam bahasa Indonesia. Dataset terdiri dari 15 dokumen raw yang mencakup berbagai penyakit seperti alergi, anemia, asma, batuk, demam, diabetes, diare, flu, hipertensi, insomnia, kolestrol, maag, obesitas, sakit kepala, dan stres.

**Fitur Sistem:**
- **Boolean Retrieval**: Menggunakan operator logika AND, OR, NOT untuk query boolean.
- **Vector Space Model**: Menggunakan TF-IDF dan cosine similarity untuk pencarian bebas.

**Proses Pengolahan Data:**
- Preprocessing: Lowercase, remove punctuation, stemming menggunakan Sastrawi.
- Indexing: Membuat inverted index untuk Boolean Retrieval dan TF-IDF vector untuk VSM.

**Evaluasi:**
Sistem dapat dievaluasi menggunakan metrik precision, recall, dan MAP@k melalui script run_evaluation.py.
""")
