import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def run_linear_probe(model_name):
    # Base directory assumed to be the parent directory of where the script is (i.e. project root)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    embeddings_path = os.path.join(base_dir, 'embeddings', f'{model_name}_embeddings.npy')
    labels_path = os.path.join(base_dir, 'embeddings', f'{model_name}_labels.npy')
    
    # Load data
    embeddings = np.load(embeddings_path)
    labels = np.load(labels_path)
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, 
        labels, 
        test_size=0.20, 
        random_state=42, 
        stratify=labels
    )
    
    # Train Logistic Regression
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model_name, accuracy
