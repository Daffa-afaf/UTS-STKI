# TODO List for UTS-STKI Project Setup

- [x] Create data/processed/ directory and empty clean .txt files for each raw data file (alergi_clean.txt, anemia_clean.txt, etc.)
- [x] Implement src/preprocess.py with text cleaning functions (lowercase, remove punctuation, stemming using Sastrawi for Indonesian)
- [x] Implement src/boolean_retrieval.py with Boolean Retrieval model (AND/OR/NOT queries)
- [x] Implement src/vsm.py with Vector Space Model (TF-IDF + Cosine similarity)
- [x] Implement src/evaluation.py with evaluation metrics (precision, recall, MAP@k)
- [x] Create app/app.py with basic Streamlit app for querying the IR system
- [x] Create notebooks/analysis.ipynb with structure for documentation and experiments
- [x] Create requirements.txt with necessary Python libraries (streamlit, nltk, scikit-learn, pandas, numpy, Sastrawi)
