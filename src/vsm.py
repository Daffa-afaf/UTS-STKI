import os
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class VectorSpaceModel:
    def __init__(self, data_dir='data/processed/'):
        self.data_dir = data_dir
        self.documents = {}
        self.doc_ids = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.load_documents()
        self.build_model()

    def load_documents(self):
        """
        Load all cleaned documents from data/processed/
        """
        for file in os.listdir(self.data_dir):
            if file.endswith('_clean.txt'):
                doc_id = file.replace('_clean.txt', '')
                with open(os.path.join(self.data_dir, file), 'r', encoding='utf-8') as f:
                    self.documents[doc_id] = f.read()
                self.doc_ids.append(doc_id)

    def build_model(self):
        """
        Build TF-IDF model
        """
        corpus = [self.documents[doc_id] for doc_id in self.doc_ids]
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query, top_k=10):
        """
        Search using cosine similarity, return list of (doc_id, snippet, score)
        """
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get top_k results
        top_indices = similarities.argsort()[-top_k:][::-1]
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include if similarity > 0
                doc_id = self.doc_ids[idx]
                snippet = self.documents[doc_id][:100]
                results.append((doc_id, snippet, similarities[idx]))

        return results
