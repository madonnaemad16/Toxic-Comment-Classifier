import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, GRU, LSTM, Dense, Dropout

def rnn(vocab_size, embedding_dim=128, input_length=200, num_classes=6):
    """
    Creates a Simple RNN model for multi-label classification.
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        SimpleRNN(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(num_classes, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def gru(vocab_size, embedding_dim=128, input_length=200, num_classes=6):
    """
    Creates a GRU model for multi-label classification.
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        GRU(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(num_classes, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def lstm(vocab_size, embedding_dim=128, input_length=200, num_classes=6):
    """
    Creates an LSTM model for multi-label classification.
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(num_classes, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
