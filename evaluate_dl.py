import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, multilabel_confusion_matrix
import deep_learning_models
import step7_sequence_vectorization as sv
import step8_train_test_split as split
import matplotlib.pyplot as plt
import seaborn as sns
from step1_data_cleaning import clean_regex
from step2_tokenization import apply_tokenization
from step3_stop_words import remove_stopwords
from step5_lemmatization import apply_lemmatization

def preprocess_data(df):
    processed_texts = []
    for text in df['comment_text']:
        cleaned = clean_regex(text)
        tokens = apply_tokenization(cleaned)
        no_stop = remove_stopwords(tokens)
        lemmatized = apply_lemmatization(no_stop)
        processed_texts.append(" ".join(lemmatized))
    return processed_texts

def evaluate_rnn(X_train, X_test, y_train, y_test, vocab_size, label_names):
    print("\n--- EVALUATING RNN ---")
    model = deep_learning_models.rnn(vocab_size)
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1)
    predictions = model.predict(X_test)
    predictions = (predictions > 0.4).astype(int)
    print("\nRNN Classification Report:")
    print(classification_report(y_test, predictions, target_names=label_names))
    return model

def evaluate_gru(X_train, X_test, y_train, y_test, vocab_size, label_names):
    print("\n--- EVALUATING GRU ---")
    model = deep_learning_models.gru(vocab_size)
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1)
    predictions = model.predict(X_test)
    predictions = (predictions > 0.4).astype(int)
    print("\nGRU Classification Report:")
    print(classification_report(y_test, predictions, target_names=label_names))
    return model

def evaluate_lstm(X_train, X_test, y_train, y_test, vocab_size, label_names):
    print("\n--- EVALUATING LSTM ---")
    model = deep_learning_models.lstm(vocab_size)
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1)
    predictions = model.predict(X_test)
    predictions = (predictions > 0.4).astype(int)
    print("\nLSTM Classification Report:")
    print(classification_report(y_test, predictions, target_names=label_names))
    return model

def plot_confusion_matrices(model_name, y_test, predictions, label_names):
    mcm = multilabel_confusion_matrix(y_test, predictions)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Confusion Matrices for {model_name}', fontsize=16)
    for i, (label, ax) in enumerate(zip(label_names, axes.flatten())):
        sns.heatmap(mcm[i], annot=True, fmt='d', ax=ax, cmap='Blues', cbar=False)
        ax.set_title(label)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticklabels(['False', 'True'])
        ax.set_yticklabels(['False', 'True'])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'confusion_matrix_{model_name.lower()}.png')

if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv(r"..\train.csv").head(10000)
    label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    y = df[label_names].values
    processed_texts = preprocess_data(df)
    X, tokenizer = sv.apply_sequence_vectorization(processed_texts)
    vocab_size = len(tokenizer.word_index) + 1
    X_train, X_test, y_train, y_test = split.split_data(X, y)
    
    for model_name, eval_func in [("RNN", evaluate_rnn), ("GRU", evaluate_gru), ("LSTM", evaluate_lstm)]:
        model = eval_func(X_train, X_test, y_train, y_test, vocab_size, label_names)
        raw_probs = model.predict(X_test)
        predictions = (raw_probs > 0.4).astype(int)
        plot_confusion_matrices(model_name, y_test, predictions, label_names)
