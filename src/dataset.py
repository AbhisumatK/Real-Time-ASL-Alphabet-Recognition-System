import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional

class ASLLandmarkExtractor:
    def __init__(self, model_path: str = "models/hand_landmarker.task"):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def extract_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts 21 hand landmarks using Mediapipe Tasks API.
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        detection_result = self.detector.detect(mp_image)

        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            
            landmarks = []
            # Get wrist (landmark 0) for relative normalization
            wrist = hand_landmarks[0]
            
            for lm in hand_landmarks:
                landmarks.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
            
            return np.array(landmarks, dtype=np.float32)
        
        return None

class ASLDataset(Dataset):
    def __init__(self, data_path: str, labels: List[str], transform=None, landmark_extractor=None):
        self.data_path = data_path
        self.labels = labels
        self.transform = transform
        self.landmark_extractor = landmark_extractor or ASLLandmarkExtractor()
        self.samples = []
        
        self._load_samples()

    def _load_samples(self):
        """
        Expects a directory structure like:
        data_path/
            A/
                A1.jpg
                ...
            B/
                ...
        """
        for label in self.labels:
            label_dir = os.path.join(self.data_path, label)
            if not os.path.isdir(label_dir):
                continue
            
            for img_name in os.listdir(label_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(label_dir, img_name), self.labels.index(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        
        if image is None:
            # Handle broken images
            return self.__getitem__((idx + 1) % len(self))

        landmarks = self.landmark_extractor.extract_landmarks(image)
        
        if landmarks is None:
            # If no hand detected, return zeroed landmarks or handle accordingly
            # For training, we might want to skip or return a dummy
            landmarks = np.zeros(63, dtype=np.float32)
            
        return torch.tensor(landmarks, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def get_dataloader(data_path: str, labels: List[str], batch_size: int = 32, shuffle: bool = True):
    dataset = ASLDataset(data_path, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
