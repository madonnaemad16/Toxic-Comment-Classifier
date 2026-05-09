import pandas as pd
import nltk
from tqdm import tqdm # Useful for progress bars

# Pipeline Steps
from step1_data_cleaning import clean_regex
from step2_tokenization import apply_tokenization
from step3_stop_words import remove_stopwords
from step5_lemmatization import apply_lemmatization
from step7_vectorization_padding import apply_tfidf_vectorization

# Modeling Steps
from step8_train_test_split import split_data
from step9_model_training import train_model, evaluate_model

# Ensure NLTK data is ready
nltk.download('wordnet')
nltk.download('omw-1.4')

# 1. Load Data
print("Loading dataset...")
try:
    df = pd.read_csv(r"..\train.csv")
    # Using a subset for demonstration speed (you can remove .head() later)
    df = df.head(5000) 
except FileNotFoundError:
    print("Error: train.csv not found.")
    exit()

# 2. Preprocess Text (Steps 1-6)
print("\nPreprocessing comments (this may take a minute)...")
processed_texts = []
for text in tqdm(df['comment_text']):
    cleaned = clean_regex(text)
    tokens = apply_tokenization(cleaned)
    no_stop = remove_stopwords(tokens)
    lemmatized = apply_lemmatization(no_stop)
    processed_texts.append(" ".join(lemmatized))

# 3. Vectorize (Step 7)
print("\nVectorizing text with TF-IDF...")
X, vectorizer = apply_tfidf_vectorization(processed_texts)

# 4. Prepare Labels
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y = df[label_cols]

# 5. Split Data (Step 8)
print("\nSplitting data into Train/Test sets...")
X_train, X_test, y_train, y_test = split_data(X, y)

# 6. Train Model (Step 9)
model = train_model(X_train, y_train)

# 7. Evaluate Model
evaluate_model(model, X_test, y_test, label_names=label_cols)

print("\n--- DONE ---")
print("Your baseline Logistic Regression model is trained and evaluated!")
