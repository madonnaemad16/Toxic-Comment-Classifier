import nltk

# Ensure required NLTK resources are downloaded
# nltk.download('averaged_perceptron_tagger')

def apply_pos_tagging(tokens):
    """
    Applies Part-of-Speech (POS) Tagging to a list of tokens.
    Returns a list of tuples (word, tag).
    """
    return nltk.pos_tag(tokens)
