from nltk.corpus import stopwords
import nltk

# Run this once: 
# nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

# Apply it:
# df['tokens_no_stop'] = df['tokens'].apply(remove_stopwords)
