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
