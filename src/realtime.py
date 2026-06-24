import cv2
import torch
import mediapipe as mp
import numpy as np
from src.dataset import ASLLandmarkExtractor
from src.model import get_model
from src.utils import get_asl_labels

def run_realtime(model_path: str, task_model_path: str = "models/hand_landmarker.task", use_gui: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = get_asl_labels()
    
    # Initialize extractor and model
    extractor = ASLLandmarkExtractor(model_path=task_model_path)
    model = get_model(num_classes=len(labels), device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
            
        # Flip image for selfie view
        image = cv2.flip(image, 1)
        
        # Extract landmarks
        landmarks = extractor.extract_landmarks(image)
        
        if landmarks is not None:
            input_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = outputs.max(1)
                confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()
                
            label = labels[predicted.item()]
            
            # Display text
            cv2.putText(image, f"Sign: {label}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Conf: {confidence:.2f}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        if use_gui:
            cv2.imshow('ASL Real-Time Recognition', image)
            if cv2.waitKey(5) & 0xFF == 27: # ESC to exit
                break
        else:
            # For Streamlit, we want to yield the image
            yield image
            
    cap.release()
    if use_gui:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # To run standalone with GUI
    # generator = run_realtime("models/asl_model.pth", use_gui=True)
    # for _ in generator: pass # Just consume the generator
    pass
