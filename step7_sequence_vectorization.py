import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def apply_sequence_vectorization(texts, max_words=20000, max_len=200):
    """
    Converts a list of preprocessed text strings into sequences of padded integers.
    
    Args:
        texts (list of str): The cleaned and preprocessed comment strings.
        max_words (int): Maximum number of words to keep in the vocabulary.
        max_len (int): Maximum length of each sequence (padding/truncating).
        
    Returns:
        sequences (ndarray): The numerical representation of your text as sequences.
        tokenizer (Tokenizer object): The fitted tokenizer.
    """
    # 1. Initialize the Tokenizer
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    
    # 2. Fit the tokenizer on the texts
    tokenizer.fit_on_texts(texts)
    
    # 3. Convert texts to sequences of integers
    sequences = tokenizer.texts_to_sequences(texts)
    
    # 4. Pad the sequences to ensure uniform length
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    
    return padded_sequences, tokenizer
