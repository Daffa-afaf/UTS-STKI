import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.boolean_retrieval import BooleanRetrieval
from src.vsm import VectorSpaceModel

# Initialize session state
if 'model_choice' not in st.session_state:
    st.session_state.model_choice = "Boolean Retrieval"
if 'query' not in st.session_state:
    st.session_state.query = ""
if 'results' not in st.session_state:
    st.session_state.results = None
if 'expanded' not in st.session_state:
    st.session_state.expanded = {}

# Sidebar untuk navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Pencarian", "Tentang Dataset", "Evaluasi"])

if page == "Pencarian":
    st.title("Sistem Informasi Retrieval (IR) - UTS STKI")

    # Layout dengan kolom
    col1, col2 = st.columns([1, 2])

    with col1:
        # Pilih model
        model_choice = st.selectbox("Pilih Model Retrieval:", ["Boolean Retrieval", "Vector Space Model"], key="model_select")
        st.session_state.model_choice = model_choice

        if model_choice == "Boolean Retrieval":
            model = BooleanRetrieval()
            query_type = "Boolean Query (e.g., 'demam AND batuk' or 'NOT alergi')"
        else:
            model = VectorSpaceModel()
            query_type = "Query bebas (e.g., 'gejala demam')"

        query = st.text_input(f"Masukkan {query_type}:", value=st.session_state.query, key="query_input")
        st.session_state.query = query

        # Tombol contoh query
        if st.button("Contoh Query"):
            if model_choice == "Boolean Retrieval":
                st.info("Contoh: 'demam AND batuk' atau 'NOT alergi'")
            else:
                st.info("Contoh: 'gejala demam tinggi' atau 'penyakit perut'")

        if st.button("Cari"):
            if query:
                if model_choice == "Boolean Retrieval":
                    st.session_state.results = model.search(query)
                else:
                    st.session_state.results = model.search(query)
                # Reset expanded states
                st.session_state.expanded = {}
            else:
                st.error("Masukkan query terlebih dahulu.")

    with col2:
        if st.session_state.results:
            if model_choice == "Boolean Retrieval":
                results = st.session_state.results
                if results:
                    st.subheader("Hasil Pencarian:")
                    for doc_id, snippet in results:
                        expanded = st.session_state.expanded.get(doc_id, False)
                        with st.expander(f"**{doc_id}**", expanded=expanded):
                            st.write(snippet + "...")
                            if st.button(f"Lihat Lengkap {doc_id}", key=f"full_{doc_id}"):
                                st.session_state.expanded[doc_id] = True
                                st.rerun()  # Force rerun to update expander
                            if expanded:
                                st.write("**Isi Dokumen Lengkap:**")
                                st.write(model.documents[doc_id])
                                st.write("---")
                else:
                    st.write("Tidak ada dokumen yang relevan.")
            else:
                results = st.session_state.results
                if results:
                    st.subheader("Hasil Pencarian:")
                    for doc_id, snippet, score in results:
                        expanded = st.session_state.expanded.get(doc_id, False)
                        with st.expander(f"**{doc_id}** (Skor: {score:.2f})", expanded=expanded):
                            st.write(snippet + "...")
                            if st.button(f"Lihat Lengkap {doc_id}", key=f"full_{doc_id}"):
                                st.session_state.expanded[doc_id] = True
                                st.rerun()  # Force rerun to update expander
                            if expanded:
                                st.write("**Isi Dokumen Lengkap:**")
                                st.write(model.documents[doc_id])
                else:
                    st.write("Tidak ada dokumen yang relevan.")

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
    st.write("Halaman ini untuk menjalankan evaluasi model dengan precision, recall, dan MAP@k.")

    # Import evaluation
    from src.evaluation import evaluate_model

    # Inisialisasi model
    br_model = BooleanRetrieval()
    vsm_model = VectorSpaceModel()

    # Contoh queries dan ground truth
    queries = ['demam', 'batuk', 'sakit AND kepala']
    ground_truth = {
        'demam': ['demam', 'flu'],
        'batuk': ['batuk', 'asma'],
        'sakit AND kepala': ['sakit_kepala', 'stres']
    }

    # Pilih model
    eval_model_choice = st.selectbox("Pilih Model untuk Evaluasi:", ["Boolean Retrieval", "Vector Space Model"])
    if eval_model_choice == "Boolean Retrieval":
        eval_model = br_model
    else:
        eval_model = vsm_model

    if st.button("Jalankan Evaluasi"):
        st.write(f"Evaluasi untuk model: {eval_model_choice}")
        results = evaluate_model(eval_model, queries, ground_truth, k=10)

        for query in queries:
            precision = results[query]['precision']
            recall = results[query]['recall']
            ap = results[query]['ap']
            retrieved_docs = results[query]['retrieved_docs']
            st.subheader(f"Query: '{query}'")
            st.write(f"Retrieved docs: {retrieved_docs}")
            st.write(f"Precision@10: {precision:.3f}")
            st.write(f"Recall@10: {recall:.3f}")
            st.write(f"Average Precision: {ap:.3f}")
            st.write("---")

        map_k = results['MAP@k']
        st.subheader(f"MAP@k: {map_k:.3f}")

  
