import os
import cv2
import numpy as np
from src.dataset import ASLLandmarkExtractor
from src.utils import get_asl_labels, download_dataset
from tqdm import tqdm

def pre_extract_landmarks(data_path: str, output_path: str = "data/asl_landmarks.npz"):
    labels = get_asl_labels()
    extractor = ASLLandmarkExtractor()
    
    all_landmarks = []
    all_labels = []
    
    os.makedirs("data", exist_ok=True)
    
    print(f"Extracting landmarks from {data_path}...")
    
    for label_idx, label in enumerate(labels):
        label_dir = os.path.join(data_path, label)
        if not os.path.exists(label_dir):
            print(f"Warning: Folder {label_dir} not found. Skipping.")
            continue
            
        print(f"Processing class: {label}")
        img_names = os.listdir(label_dir)
        
        for img_name in tqdm(img_names):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(label_dir, img_name)
            image = cv2.imread(img_path)
            
            if image is None:
                continue
                
            landmarks = extractor.extract_landmarks(image)
            
            if landmarks is not None:
                all_landmarks.append(landmarks)
                all_labels.append(label_idx)
    
    all_landmarks = np.array(all_landmarks)
    all_labels = np.array(all_labels)
    
    print(f"Saving {len(all_landmarks)} samples to {output_path}...")
    np.savez_compressed(output_path, landmarks=all_landmarks, labels=all_labels)
    print("Done!")

if __name__ == "__main__":
    DATA_PATH = r"C:\Users\ABHISUMAT\.cache\kagglehub\datasets\grassknoted\asl-alphabet\versions\1\asl_alphabet_train\asl_alphabet_train"
    pre_extract_landmarks(DATA_PATH)
