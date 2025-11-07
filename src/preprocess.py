import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def preprocess_file(input_file, output_file):
    """
    Preprocess a text file: lowercase, remove punctuation, stem using Sastrawi.
    """
    stemmer = StemmerFactory().create_stemmer()

    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Split into words, stem each, join back
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    cleaned_text = ' '.join(stemmed_words)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)

# Example usage (can be removed later)
# preprocess_file('data/raw/demam.txt', 'data/processed/demam_clean.txt')
