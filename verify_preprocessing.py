import nltk
import pandas as pd
from step1_data_cleaning import clean_regex
from step2_tokenization import apply_tokenization
from step3_stop_words import remove_stopwords
from step4_stemming import apply_stemming
from step5_lemmatization import apply_lemmatization
from step6_pos_tagging import apply_pos_tagging
from step10_ner import apply_ner
import step7_sequence_vectorization as sv

# Download necessary resources
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('maxent_ne_chunker_tab', quiet=True)
nltk.download('words', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt', quiet=True)

def verify_pipeline(sample_text):
    print(f"Original Text: {sample_text}")
    
    # Step 1: Cleaning
    cleaned = clean_regex(sample_text)
    print(f"\nStep 1 (Cleaning): {cleaned}")
    
    # Step 2: Tokenization
    tokens = apply_tokenization(cleaned)
    print(f"\nStep 2 (Tokenization): {tokens}")
    
    # Step 3: Stop Words
    no_stop = remove_stopwords(tokens)
    print(f"\nStep 3 (Stop Words): {no_stop}")
    
    # Step 4: Stemming
    stemmed = apply_stemming(no_stop)
    print(f"\nStep 4 (Stemming): {stemmed}")
    
    # Step 5: Lemmatization
    lemmatized = apply_lemmatization(no_stop) # Using original tokens usually better for lemma
    print(f"\nStep 5 (Lemmatization): {lemmatized}")
    
    # Step 6: POS Tagging
    pos_tags = apply_pos_tagging(lemmatized)
    print(f"\nStep 6 (POS Tagging): {pos_tags}")
    
    # Step 10: NER (requires preserved case usually)
    tokens_with_case = apply_tokenization(sample_text)
    ner_tree = apply_ner(tokens_with_case)
    print(f"\nStep 10 (NER):")
    # Pretty print the tree structure for entities
    entities_found = False
    for chunk in ner_tree:
        if hasattr(chunk, 'label'):
            print(f"  Entity Found: {' '.join(c[0] for c in chunk)} ({chunk.label()})")
            entities_found = True
    if not entities_found:
        print("  No entities found.")
    
    # Step 7: Sequence Vectorization
    processed_string = " ".join(lemmatized)
    X, tokenizer = sv.apply_sequence_vectorization([processed_string])
    print(f"\nStep 7 (Vectorization - Sequence): {X}")

if __name__ == "__main__":
    test_comment = "Hello! I am a software engineer working at Google in New York City. This is a toxic comment!"
    verify_pipeline(test_comment)
