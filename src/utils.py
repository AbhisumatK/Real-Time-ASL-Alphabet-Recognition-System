import kagglehub
import os
import shutil

def download_dataset():
    """
    Downloads the ASL Alphabet dataset from Kaggle using kagglehub.
    """
    print("Downloading dataset...")
    path = kagglehub.dataset_download("grassknoted/asl-alphabet")
    
    print(f"Dataset downloaded to: {path}")
    
    # We might want to move it to our data/ directory or just use the path
    return path

def get_asl_labels():
    """
    Returns the list of ASL labels.
    """
    return [chr(i) for i in range(ord('A'), ord('Z') + 1)] + ['del', 'nothing', 'space']

if __name__ == "__main__":
    download_dataset()
