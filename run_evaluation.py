import sys
sys.path.append('.')  # Tambahkan path agar bisa import dari src

from src.evaluation import evaluate_model
from src.boolean_retrieval import BooleanRetrieval  # Menggunakan BooleanRetrieval
from src.vsm import VectorSpaceModel

# Inisialisasi model
br_model = BooleanRetrieval()
vsm_model = VectorSpaceModel()

# Contoh queries dan ground truth
queries = ['demam', 'batuk', 'flu']  # List query
ground_truth = {
    'demam': ['demam', 'flu'],
    'batuk': ['batuk', 'asma'],
    'flu': ['flu', 'demam']
}  # Dict {query: list relevant doc_ids}

# Jalankan evaluasi untuk Boolean Retrieval
print("Evaluasi Boolean Retrieval:")
br_results = evaluate_model(br_model, queries, ground_truth, k=10)
print(br_results)

# Jalankan evaluasi untuk VSM
print("\nEvaluasi VSM:")
vsm_results = evaluate_model(vsm_model, queries, ground_truth, k=10)
print(vsm_results)
