"""
Text preprocessing utilities for sentiment analysis.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


def clean_text(text, remove_stopwords=True, stem_words=False):
    """
    Clean and preprocess text data.
    
    Args:
        text (str): Input text to clean
        remove_stopwords (bool): Whether to remove stopwords
        stem_words (bool): Whether to apply stemming
        
    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    if stem_words:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
    
    # Join tokens back
    text = ' '.join(tokens)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def preprocess_dataset(dataset, text_column='text', remove_stopwords=True, stem_words=False):
    """
    Preprocess all texts in a dataset.
    
    Args:
        dataset: Hugging Face dataset
        text_column (str): Name of the text column
        remove_stopwords (bool): Whether to remove stopwords
        stem_words (bool): Whether to apply stemming
        
    Returns:
        Dataset: Preprocessed dataset
    """
    def clean_example(example):
        example[text_column] = clean_text(
            example[text_column],
            remove_stopwords=remove_stopwords,
            stem_words=stem_words
        )
        return example
    
    return dataset.map(clean_example)


if __name__ == "__main__":
    # Test preprocessing
    sample_text = "This movie is absolutely fantastic! I loved every moment of it."
    cleaned = clean_text(sample_text)
    print(f"Original: {sample_text}")
    print(f"Cleaned: {cleaned}")

