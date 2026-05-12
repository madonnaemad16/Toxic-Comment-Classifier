import nltk
from nltk import ne_chunk, pos_tag, word_tokenize

# nltk.download('maxent_ne_chunker')
# nltk.download('words')

def apply_ner(tokens):
    """
    Applies Named Entity Recognition (NER) to a list of tokens.
    Requires POS tags as intermediate input.
    Returns a tree structure (nltk.tree.Tree) or list of chunks.
    """
    pos_tags = pos_tag(tokens)
    ner_chunks = ne_chunk(pos_tags)
    return ner_chunks
