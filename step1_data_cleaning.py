import re

def clean_regex(text):
    text = str(text).lower()
    # Remove HTML tags (e.g. <br>)
    text = re.sub(r'<.*?>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove special characters and numbers (keeping only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Apply it:
# df['step1_regex'] = df['comment_text'].apply(clean_regex)
# print("Original:\n", df['comment_text'].iloc[0])
# print("\nAfter Regex:\n", df['step1_regex'].iloc[0])
