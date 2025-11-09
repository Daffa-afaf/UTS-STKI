import sys
sys.path.append('.')

from src.evaluation import evaluate_model
from src.boolean_retrieval import BooleanRetrieval
from src.vsm import VectorSpaceModel

br = BooleanRetrieval()
vsm = VectorSpaceModel(data_dir='data/processed')

queries = ['demam', 'batuk', 'sakit kepala', 'flu', 'asma', 'diabetes']
ground_truth = {
    'demam': ['demam', 'flu'],
    'batuk': ['batuk', 'asma'],
    'sakit kepala': ['sakit_kepala', 'stres'],
    'flu': ['flu', 'demam'],
    'asma': ['asma', 'batuk'],
    'diabetes': ['diabetes']
}

br_results = evaluate_model(br, queries, ground_truth, k=10)
vsm_results = evaluate_model(vsm, queries, ground_truth, k=10)

print("Evaluasi Boolean Retrieval:")
for query, metrics in br_results.items():
    if query != 'MAP@k':
        print(f"{query}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, AP={metrics['ap']:.3f}")
print(f"MAP@k: {br_results['MAP@k']:.3f}")

print("\nEvaluasi VSM:")
for query, metrics in vsm_results.items():
    if query != 'MAP@k':
        print(f"{query}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, AP={metrics['ap']:.3f}")
print(f"MAP@k: {vsm_results['MAP@k']:.3f}")
