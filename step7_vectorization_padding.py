import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def apply_tfidf_vectorization(texts, max_features=5000):
    """
    Converts a list of preprocessed text strings into a TF-IDF matrix.
    This follows the 'Text Representation' PPT using scikit-learn.
    
    Args:
        texts (list of str): The cleaned and preprocessed comment strings.
        max_features (int): Maximum number of top words to keep.
        
    Returns:
        tfidf_matrix (sparse matrix): The numerical representation of your text.
        vectorizer (TfidfVectorizer object): The fitted vectorizer.
    """
    # 1. Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=max_features)
    
    # 2. Fit and transform the texts
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    return tfidf_matrix, vectorizer
