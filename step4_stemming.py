from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer("english")

def apply_stemming(tokens):
    return [stemmer.stem(word) for word in tokens]

# Apply it:
# df['stemmed_tokens'] = df['tokens_no_stop'].apply(apply_stemming)
