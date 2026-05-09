import pandas as pd
import nltk
import numpy as np

# Import all 7 steps
from step1_data_cleaning import clean_regex
from step2_tokenization import apply_tokenization
from step3_stop_words import remove_stopwords
from step4_stemming import apply_stemming
from step5_lemmatization import apply_lemmatization
from step6_pos_tagging import apply_pos_tagging
from step7_vectorization_padding import apply_tfidf_vectorization

# Download NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')

print("\nLoading data...")
try:
    df = pd.read_csv(r"..\train.csv").head(5) # Sample of 5 rows
except FileNotFoundError:
    print("Error: train.csv not found.")
    exit()

print("\n--- RUNNING STEPS 1-6 (Text Preprocessing) ---")

processed_texts = []

for i, row in df.iterrows():
    text = row['comment_text']
    
    # Preprocessing
    cleaned = clean_regex(text)
    tokens = apply_tokenization(cleaned)
    no_stop = remove_stopwords(tokens)
    lemmatized = apply_lemmatization(no_stop)
    
    # Join tokens back into a single string for TF-IDF
    processed_string = " ".join(lemmatized)
    processed_texts.append(processed_string)
    
    print(f"\nComment {i+1} Processed: {processed_string[:100]}...")

print("\n--- RUNNING STEP 7 (TF-IDF Vectorization) ---")

# Apply Step 7
tfidf_matrix, vectorizer = apply_tfidf_vectorization(processed_texts, max_features=100)

print("\nTF-IDF Matrix Shape:", tfidf_matrix.shape)
print("Vocabulary (first 10 words):", vectorizer.get_feature_names_out()[:10])

# Show a sample of the matrix
first_row_dense = tfidf_matrix[0].todense()
print("\nTF-IDF Weights for first comment (non-zero entries):")
for i, weight in enumerate(np.asarray(first_row_dense)[0]):
    if weight > 0:
        print(f"Word: {vectorizer.get_feature_names_out()[i]} -> Weight: {weight:.4f}")

print("\n--- FULL SYSTEM TEST COMPLETE ---")
print("Your data is now ready for a Machine Learning Classifier!")
