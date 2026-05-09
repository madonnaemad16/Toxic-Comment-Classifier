from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the vectorized data (X) and labels (y) into Training and Testing sets.
    
    Args:
        X: The vectorized text matrix (e.g., TF-IDF matrix).
        y: The labels (target columns).
        test_size (float): Percentage of data to use for testing (default 20%).
        random_state (int): Seed for reproducibility.
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test
