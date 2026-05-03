from nltk.tokenize import RegexpTokenizer

# Best practice: extract only alphanumeric words
tokenizer = RegexpTokenizer(r'\w+')

def apply_tokenization(text):
    return tokenizer.tokenize(text)

# Apply it:
# df['tokens'] = df['step1_regex'].apply(apply_tokenization)
