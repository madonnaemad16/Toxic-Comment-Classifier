import pandas as pd

# Import the functions from our 4 steps
from step1_data_cleaning import clean_regex
from step2_tokenization import apply_tokenization
from step3_stop_words import remove_stopwords
from step4_stemming import apply_stemming

print("Loading data...")

# ---------------------------------------------------------
# OPTION 1: Using your specific dataset
# Uncomment the line below and put the path to your CSV file
df = pd.read_csv(r"..\train.csv").head(10) # Just testing on the first 10 rows
# ---------------------------------------------------------

# ---------------------------------------------------------
# OPTION 2: Using dummy test data (we will use this for now)
# ---------------------------------------------------------
#data = {
 #   'comment_text': [
  #      "You are an IDIOT!!! I hate you >:( http://badsite.com",
   #     "Please read the Wikipedia article <br> it is very helpful.",
    #    "What the hell are you talking about? 12345 STOP!!"
    #]
#}
# df = pd.DataFrame(data)

print("\n--- ORIGINAL DATA ---")
for comment in df['comment_text']:
    print(f"- {comment}")

# 1. Apply Step 1: Cleaning
df['step1_clean'] = df['comment_text'].apply(clean_regex)
print("\n--- AFTER STEP 1 (Regex Clean) ---")
for text in df['step1_clean']:
    print(f"- {text}")

# 2. Apply Step 2: Tokenization
df['step2_tokens'] = df['step1_clean'].apply(apply_tokenization)
print("\n--- AFTER STEP 2 (Tokenization) ---")
for tokens in df['step2_tokens']:
    print(f"- {tokens}")

# 3. Apply Step 3: Stop Words Removal
df['step3_no_stop'] = df['step2_tokens'].apply(remove_stopwords)
print("\n--- AFTER STEP 3 (Stop Words Removed) ---")
for tokens in df['step3_no_stop']:
    print(f"- {tokens}")

# 4. Apply Step 4: Stemming
df['step4_stemmed'] = df['step3_no_stop'].apply(apply_stemming)
print("\n--- AFTER STEP 4 (Stemmed) ---")
for tokens in df['step4_stemmed']:
    print(f"- {tokens}")

print("\nPipeline Test Complete!")
