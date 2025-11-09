# Information Retrieval using TF-IDF & Vector Space Model

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Finished-brightgreen)
![Made with](https://img.shields.io/badge/Made%20with-scikit--learn-orange)
![Repo Size](https://img.shields.io/github/repo-size/username/UTS-STKI?color=blueviolet)

---

## Deskripsi

Proyek ini merupakan implementasi **Information Retrieval System (IRS)** berbasis:
- **TF-IDF (Term Frequency–Inverse Document Frequency)**
- **Vector Space Model (VSM)**
- serta pembanding **Boolean Retrieval Model**

Sistem digunakan untuk mencari dokumen relevan berdasarkan *query pengguna* dan dievaluasi menggunakan metrik:
- Precision@K  
- Recall@K  
- Average Precision (AP)  
- Mean Average Precision (MAP@K)

Eksperimen ini merupakan bagian dari tugas **Ujian Tengah Semester (UTS) Sistem Temu Kembali Informasi (STKI)**.


# Struktur Proyek
UTS-STKI/
├── app/
│ └── app.py # Aplikasi utama untuk menjalankan pencarian dokumen
│
├── data/
│ ├── processed/ # Dokumen hasil preprocessing
│ └── raw/ # Dokumen mentah
│
├── notebooks/
│ └── analysis.ipynb # Notebook untuk eksplorasi dan visualisasi
│
├── src/
│ ├── boolean_retrieval.py # Implementasi model boolean retrieval
│ ├── evaluation.py # Perhitungan metrik evaluasi IR
│ ├── preprocess.py # Pembersihan dan normalisasi teks
│ ├── vsm.py # Implementasi Vector Space Model (TF-IDF)
│ ├── run_evaluation.py # Menjalankan evaluasi keseluruhan model
│ ├── requirements.txt # Dependensi Python
│ ├── AP.png # Grafik Average Precision
│ ├── MAP@K.png
 # Grafik Mean Average Precision
│ ├── recall.png # Grafik Recall@K
│ ├── precession.png # Grafik Precision@K
│ ├── Heatmap.png # Visualisasi hubungan dokumen-query
│ ├── Trend.png # Tren performa hasil pencarian
│ └── tf-idf.png # Visualisasi Top 10 TF-IDF Terms
