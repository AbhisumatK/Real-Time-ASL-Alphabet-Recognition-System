import torch
import cv2
import numpy as np
from src.dataset import ASLLandmarkExtractor
from src.model import get_model
from src.utils import get_asl_labels

def predict_single_image(image_path: str, model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = get_asl_labels()
    
    extractor = ASLLandmarkExtractor()
    model = get_model(num_classes=len(labels), device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
        
    landmarks = extractor.extract_landmarks(image)
    
    if landmarks is not None:
        input_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = outputs.max(1)
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()
            
        label = labels[predicted.item()]
        print(f"Predicted: {label} (Confidence: {confidence:.2f})")
        return label, confidence
    else:
        print("No hand detected in the image.")
        return None, 0.0

if __name__ == "__main__":
    # predict_single_image("sample.jpg", "models/asl_model.pth")
    pass
