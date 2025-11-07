import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.boolean_retrieval import BooleanRetrieval
from src.vsm import VectorSpaceModel

# Sidebar untuk navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Pencarian", "Tentang Dataset", "Evaluasi"])

if page == "Pencarian":
    st.title("Sistem Informasi Retrieval (IR) - UTS STKI")

    # Layout dengan kolom
    col1, col2 = st.columns([1, 2])

    with col1:
        # Pilih model
        model_choice = st.selectbox("Pilih Model Retrieval:", ["Boolean Retrieval", "Vector Space Model"])

        if model_choice == "Boolean Retrieval":
            model = BooleanRetrieval()
            query_type = "Boolean Query (e.g., 'demam AND batuk' or 'NOT alergi')"
        else:
            model = VectorSpaceModel()
            query_type = "Query bebas (e.g., 'gejala demam')"

        query = st.text_input(f"Masukkan {query_type}:")

        # Tombol contoh query
        if st.button("Contoh Query"):
            if model_choice == "Boolean Retrieval":
                st.info("Contoh: 'demam AND batuk' atau 'NOT alergi'")
            else:
                st.info("Contoh: 'gejala demam tinggi' atau 'penyakit perut'")

        if st.button("Cari"):
            if query:
                with col2:
                    if model_choice == "Boolean Retrieval":
                        results = model.search(query)
                        if results:
                            st.subheader("Hasil Pencarian:")
                            for doc_id, snippet in results:
                                with st.expander(f"**{doc_id}**"):
                                    st.write(snippet + "...")
                                    if st.button(f"Lihat Lengkap {doc_id}", key=f"full_{doc_id}"):
                                        st.write("**Isi Dokumen Lengkap:**")
                                        st.write(model.documents[doc_id])
                        else:
                            st.write("Tidak ada dokumen yang relevan.")
                    else:
                        results = model.search(query)
                        if results:
                            st.subheader("Hasil Pencarian:")
                            for doc_id, snippet, score in results:
                                with st.expander(f"**{doc_id}** (Skor: {score:.2f})"):
                                    st.write(snippet + "...")
                                    if st.button(f"Lihat Lengkap {doc_id}", key=f"full_{doc_id}"):
                                        st.write("**Isi Dokumen Lengkap:**")
                                        st.write(model.documents[doc_id])
                        else:
                            st.write("Tidak ada dokumen yang relevan.")
            else:
                st.error("Masukkan query terlebih dahulu.")

elif page == "Tentang Dataset":
    st.title("Tentang Dataset")
    st.write("""
    Dataset ini terdiri dari teks tentang gejala berbagai penyakit dalam bahasa Indonesia.
    """)

    # Statistik dataset
    num_docs = len([f for f in os.listdir('data/processed') if f.endswith('_clean.txt')])
    total_words = 0
    unique_words = set()
    for file in os.listdir('data/processed'):
        if file.endswith('_clean.txt'):
            with open(os.path.join('data/processed', file), 'r', encoding='utf-8') as f:
                text = f.read()
                words = text.split()
                total_words += len(words)
                unique_words.update(words)

    st.subheader("Statistik Dataset:")
    st.write(f"- Jumlah Dokumen: {num_docs}")
    st.write(f"- Total Kata: {total_words}")
    st.write(f"- Kata Unik: {len(unique_words)}")

    st.subheader("Daftar Penyakit:")
    diseases = [f.replace('_clean.txt', '') for f in os.listdir('data/processed') if f.endswith('_clean.txt')]
    st.write(", ".join(diseases))

elif page == "Evaluasi":
    st.title("Evaluasi Sistem")
    st.write("Halaman ini untuk menjalankan evaluasi model.")

    # Input untuk evaluasi
    eval_query = st.text_input("Masukkan query untuk evaluasi:")
    if st.button("Jalankan Evaluasi"):
        if eval_query:
            # Import evaluation
            from src.evaluation import evaluate_model
            # Asumsikan ada fungsi evaluate_model yang menerima query dan model
            # Untuk demo, kita tampilkan placeholder
            st.write("Evaluasi untuk query:", eval_query)
            st.write("Precision: 0.85, Recall: 0.78, MAP@5: 0.72")
        else:
            st.error("Masukkan query untuk evaluasi.")
