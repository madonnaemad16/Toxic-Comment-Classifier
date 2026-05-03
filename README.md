# Toxic Comment Classifier

> An end-to-end Natural Language Processing (NLP) pipeline and LSTM neural network designed to automatically detect and classify toxic online text.

## 📌 Project Overview
This project implements an advanced NLP system to identify and classify toxic behavior in online comments (e.g., toxicity, severe toxicity, obscenity, threats, insults, and identity hate). It leverages a deep learning approach using Long Short-Term Memory (LSTM) networks to capture sequential text dependencies, ensuring high accuracy in sentiment and toxicity analysis.

This was developed as the final project for the **Selected Topics in Data Science** course, demonstrating proficiency in end-to-end machine learning engineering, from raw text preprocessing to model evaluation.

## ⚙️ Text Preprocessing Pipeline
The raw text data undergoes a rigorous 7-step preprocessing pipeline to ensure maximum signal quality before being fed into the neural network:

1. **Regular Expressions (Regex):** Used `re` to clean noise, including HTML tags and URLs.
2. **Tokenization:** Applied `RegexpTokenizer` to split sentences into individual word tokens.
3. **Stop Words Removal:** Filtered out common English stop words using the `nltk` corpus to reduce dimensionality.
4. **Stemming:** Reduced words to their root forms using the Snowball Stemmer.
5. **Lemmatization:** Mapped words to their dictionary forms (lemmas) for better semantic understanding using `WordNetLemmatizer`.
6. **Part-of-Speech (POS) Tagging:** Annotated tokens with their grammatical parts of speech to retain context.
7. **Sequence Padding:** Standardized input lengths for the LSTM using Keras `pad_sequences`.

## 🧠 Model Architecture
- **Deep Learning Model:** Long Short-Term Memory (LSTM) Neural Network.
- **Why LSTM?** LSTMs are exceptionally well-suited for text classification as they maintain a memory of previous words in a sentence, allowing the model to understand context and sequential meaning better than standard feedforward networks.
- **Frameworks Used:** TensorFlow / Keras

## 📊 Evaluation Metrics
The model's performance is rigorously evaluated using standard classification metrics:
- **Accuracy:** Overall correctness of the model's predictions.
- **Confusion Matrix:** Detailed breakdown of True Positives, True Negatives, False Positives, and False Negatives to evaluate the model's precision and recall across different toxicity classes.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook / Kaggle Notebooks
- Libraries: `numpy`, `pandas`, `nltk`, `scikit-learn`, `tensorflow`, `keras`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Toxic-Comment-Classifier.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the NLTK data:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')
   ```

## 📂 Dataset
The model is trained on the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset. 


