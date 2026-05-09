import pandas as pd
import nltk

# Import all steps
from step1_data_cleaning import clean_regex
from step2_tokenization import apply_tokenization
from step3_stop_words import remove_stopwords
from step4_stemming import apply_stemming
from step5_lemmatization import apply_lemmatization
from step6_pos_tagging import apply_pos_tagging

# Download necessary NLTK data for the new steps
print("Downloading NLTK data...")
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')

print("\nLoading data...")
try:
    # Testing on the first 5 rows for clarity
    df = pd.read_csv(r"..\train.csv").head(5) 
except FileNotFoundError:
    print("Error: train.csv not found in parent directory. Please check the path.")
    exit()

print("\n--- STARTING FULL PIPELINE TEST ---")

for i, row in df.iterrows():
    print(f"\n>>> PROCESSING COMMENT {i+1} <<<")
    original = row['comment_text']
    print(f"Original: {original[:100]}..." if len(original) > 100 else f"Original: {original}")

    # 1. Clean
    cleaned = clean_regex(original)
    
    # 2. Tokenize
    tokens = apply_tokenization(cleaned)
    
    # 3. Stop words
    no_stop = remove_stopwords(tokens)
    
    # 4. Stemming
    stemmed = apply_stemming(no_stop)
    
    # 5. Lemmatization (using no_stop tokens as base for better comparison)
    lemmatized = apply_lemmatization(no_stop)
    
    # 6. POS Tagging (using lemmatized tokens)
    pos_tags = apply_pos_tagging(lemmatized)

    print(f"Tokens (Step 2): {tokens}")
    print(f"No Stopwords (Step 3): {no_stop}")
    print(f"Stemmed (Step 4): {stemmed}")
    print(f"Lemmatized (Step 5): {lemmatized}")
    print(f"POS Tags (Step 6): {pos_tags}")

print("\n--- FULL PIPELINE TEST COMPLETE ---")
