import nltk
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK resources are downloaded
# nltk.download('wordnet')
# nltk.download('omw-1.4')

def apply_lemmatization(tokens):
    """
    Applies Lemmatization to a list of tokens using WordNetLemmatizer.
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]
