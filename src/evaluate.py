import torch
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from src.dataset import ASLDataset
from src.model import get_model
from src.utils import get_asl_labels
from torch.utils.data import DataLoader

def evaluate_model(data_path: str, model_path: str, npz_path: str = "data/asl_landmarks.npz"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = get_asl_labels()
    
    if os.path.exists(npz_path):
        print(f"Evaluating on pre-extracted landmarks from {npz_path}...")
        from src.landmark_dataset import LandmarkDataset
        dataset = LandmarkDataset(npz_path)
    else:
        print(f"Evaluating on raw images from {data_path}...")
        dataset = ASLDataset(data_path, labels)
        
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    model = get_model(num_classes=len(labels), device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            
    print("Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=labels))
    
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('models/confusion_matrix.png')
    plt.show()

if __name__ == "__main__":
    TEST_DATA_PATH = r"C:\Users\ABHISUMAT\.cache\kagglehub\datasets\grassknoted\asl-alphabet\versions\1\asl_alphabet_train\asl_alphabet_train"
    MODEL_PATH = "models/asl_model.pth"
    
    # We use a validation subset or just evaluate on a portion of the train for now
    # as the test folder structure is different (flat list of images).
    # To evaluate properly on the test set, we would need a customized loader.
    # For now, let's verify on the main data.
    evaluate_model(data_path=TEST_DATA_PATH, model_path=MODEL_PATH)
