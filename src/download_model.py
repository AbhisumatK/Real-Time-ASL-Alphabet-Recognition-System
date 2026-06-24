import urllib.request
import os

def download_mediapipe_model():
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    dest = "models/hand_landmarker.task"
    
    if not os.path.exists("models"):
        os.makedirs("models")
        
    if not os.path.exists(dest):
        print(f"Downloading MediaPipe model from {url}...")
        urllib.request.urlretrieve(url, dest)
        print(f"Downloaded to {dest}")
    else:
        print("Model already exists.")

if __name__ == "__main__":
    download_mediapipe_model()
