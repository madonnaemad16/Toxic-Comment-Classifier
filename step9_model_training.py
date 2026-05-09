from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model(X_train, y_train):
    """
    Trains a Logistic Regression model for multi-label classification.
    """
    # Initialize a baseline Logistic Regression model
    log_reg = LogisticRegression(solver='liblinear', random_state=42)
    
    # Use MultiOutputClassifier since we have multiple labels (toxic, severe_toxic, etc.)
    model = MultiOutputClassifier(log_reg)
    
    print("Training model (this may take a moment)...")
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test, label_names):
    """
    Evaluates the model on the test set and prints a classification report.
    """
    predictions = model.predict(X_test)
    
    print("\n--- MODEL PERFORMANCE ---")
    print(f"Overall Accuracy: {accuracy_score(y_test, predictions):.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, predictions, target_names=label_names))
    
    return predictions
