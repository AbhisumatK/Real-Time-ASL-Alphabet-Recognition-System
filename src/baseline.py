from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from src.dataset import ASLDataset
from src.utils import get_asl_labels
import torch

def train_baseline(data_path: str):
    labels = get_asl_labels()
    dataset = ASLDataset(data_path, labels)
    
    X = []
    y = []
    
    print("Extracting landmarks for baseline model...")
    for i in range(len(dataset)):
        landmarks, label = dataset[i]
        X.append(landmarks.numpy())
        y.append(label.item())
        
    X = np.array(X)
    y = np.array(y)
    
    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    preds = rf.predict(X_test)
    print(f"Random Forest Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, target_names=labels))
    
    return rf

if __name__ == "__main__":
    # train_baseline("path/to/data")
    pass
