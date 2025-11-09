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
        query_lower = query.lower()
        tokens = re.split(r'(\s+)', query_lower)
        tokens = [t.strip() for t in tokens if t.strip()]

        # Check for operators
        if 'and' in tokens:
            idx = tokens.index('and')
            left = ' '.join(tokens[:idx])
            right = ' '.join(tokens[idx+1:])
            return 'AND', left, right
        elif 'or' in tokens:
            idx = tokens.index('or')
            left = ' '.join(tokens[:idx])
            right = ' '.join(tokens[idx+1:])
            return 'OR', left, right
        elif 'not' in tokens:
            idx = tokens.index('not')
            term = ' '.join(tokens[idx+1:])
            return 'NOT', term, None
        else:
            # No operators, split into terms and assume AND for multiple terms
            terms = query_lower.split()
            if len(terms) == 1:
                return 'TERM', terms[0], None
            else:
                # Assume AND for multiple terms
                return 'AND', terms[0], ' '.join(terms[1:])

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

        result_docs = sorted(result_docs, key=len)
        results = []
        for doc_id in result_docs:
            snippet = self.documents[doc_id][:100]  # 100 karakter pertama sebagai cuplikan
            results.append((doc_id, snippet))

        return results
