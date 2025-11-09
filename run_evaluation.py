import sys
sys.path.append('.')  # Tambahkan path agar bisa import dari src

from src.evaluation import evaluate_model
from src.boolean_retrieval import BooleanRetrieval  # Menggunakan BooleanRetrieval

# Inisialisasi model Boolean Retrieval
br_model = BooleanRetrieval()

# 3 Query baru beserta gold relevant docs
queries = ['demam', 'batuk', 'sakit AND kepala']  # List query (gunakan AND untuk query multi-term)
ground_truth = {
    'demam': ['demam', 'flu'],  # Gold relevant docs untuk query 'demam'
    'batuk': ['batuk', 'asma'],  # Gold relevant docs untuk query 'batuk'
    'sakit AND kepala': ['sakit_kepala', 'stres']  # Gold relevant docs untuk query 'sakit AND kepala'
}  # Dict {query: list relevant doc_ids}

# Jalankan evaluasi untuk Boolean Retrieval
print("Evaluasi Boolean Retrieval untuk 3 Query:")
br_results = evaluate_model(br_model, queries, ground_truth, k=10)

# Tampilkan hasil precision dan recall untuk setiap query
for query in queries:
    precision = br_results[query]['precision']
    recall = br_results[query]['recall']
    retrieved_docs = br_results[query]['retrieved_docs']
    print(f"Query: '{query}'")
    print(f"  Retrieved docs: {retrieved_docs}")
    print(f"  Precision@10: {precision:.3f}")
    print(f"  Recall@10: {recall:.3f}")
    print(f"  Average Precision: {br_results[query]['ap']:.3f}")
    print()

print(f"MAP@k: {br_results['MAP@k']:.3f}")
