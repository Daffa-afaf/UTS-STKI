import os
import re

class BooleanRetrieval:
    def __init__(self, data_dir='data/processed/'):
        self.data_dir = data_dir
        self.documents = {}
        self.index = {}
        self.load_documents()
        self.build_index()

    def load_documents(self):
        """
        Load all cleaned documents from data/processed/
        """
        for file in os.listdir(self.data_dir):
            if file.endswith('_clean.txt'):
                doc_id = file.replace('_clean.txt', '')
                with open(os.path.join(self.data_dir, file), 'r', encoding='utf-8') as f:
                    self.documents[doc_id] = f.read()

    def build_index(self):
        """
        Build inverted index: term -> set of doc_ids
        """
        for doc_id, content in self.documents.items():
            terms = set(content.split())
            for term in terms:
                if term not in self.index:
                    self.index[term] = set()
                self.index[term].add(doc_id)

    def parse_query(self, query):
        """
        Parse boolean query: support AND, OR, NOT
        Simple parser for terms separated by operators.
        """
        # Split by spaces, but keep operators
        tokens = re.split(r'(\s+)', query)
        tokens = [t.strip() for t in tokens if t.strip()]

        # For simplicity, assume query like "term1 AND term2" or "term1 OR term2" or "NOT term"
        # This is a basic implementation; can be extended for complex queries.

        if 'AND' in tokens:
            idx = tokens.index('AND')
            left = tokens[:idx]
            right = tokens[idx+1:]
            left_term = ' '.join(left).strip()
            right_term = ' '.join(right).strip()
            return 'AND', left_term, right_term
        elif 'OR' in tokens:
            idx = tokens.index('OR')
            left = tokens[:idx]
            right = tokens[idx+1:]
            left_term = ' '.join(left).strip()
            right_term = ' '.join(right).strip()
            return 'OR', left_term, right_term
        elif 'NOT' in tokens:
            idx = tokens.index('NOT')
            term = ' '.join(tokens[idx+1:]).strip()
            return 'NOT', term, None
        else:
            # Single term
            return 'TERM', ' '.join(tokens).strip(), None

    def search(self, query):
        """
        Perform boolean search and return list of (doc_id, snippet)
        """
        op, term1, term2 = self.parse_query(query.lower())

        if op == 'AND':
            set1 = self.index.get(term1, set())
            set2 = self.index.get(term2, set())
            result_docs = set1 & set2
        elif op == 'OR':
            set1 = self.index.get(term1, set())
            set2 = self.index.get(term2, set())
            result_docs = set1 | set2
        elif op == 'NOT':
            all_docs = set(self.documents.keys())
            set1 = self.index.get(term1, set())
            result_docs = all_docs - set1
        else:
            result_docs = self.index.get(term1, set())

        results = []
        for doc_id in result_docs:
            snippet = self.documents[doc_id][:100]  # First 100 chars as snippet
            results.append((doc_id, snippet))

        return results
