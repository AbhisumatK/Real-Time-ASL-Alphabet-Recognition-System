import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
from src.dataset import ASLLandmarkExtractor
from src.model import get_model
from src.utils import get_asl_labels
import os

st.set_page_config(page_title="ASL Alphabet Recognition", layout="wide")

st.title("🤟 Real-Time ASL Alphabet Recognition")
st.markdown("""
Extract hand landmarks using MediaPipe and classify ASL signs using a Deep Neural Network.
""")

# Load Model
@st.cache_resource
def load_asl_model():
    model_path = "models/asl_model.pth"
    labels = get_asl_labels()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(model_path):
        st.error("Model file not found! Please train the model first.")
        return None, None, None
        
    model = get_model(num_classes=len(labels), device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, labels, device

model, labels, device = load_asl_model()
extractor = ASLLandmarkExtractor()

# Sidebar
st.sidebar.title("Settings")
mode = st.sidebar.selectbox("Choose Input Mode", ["Webcam", "Image Upload"])

if "history" not in st.session_state:
    st.session_state.history = []

if model:
    if mode == "Image Upload":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Convert PIL to OpenCV
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            landmarks = extractor.extract_landmarks(cv_image)
            
            if landmarks is not None:
                input_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, predicted = outputs.max(1)
                    confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()
                
                label = labels[predicted.item()]
                st.success(f"Prediction: **{label}**")
                st.write(f"Confidence: **{confidence:.2f}**")
                st.session_state.history.append(label)
            else:
                st.warning("No hand detected in the image.")

    elif mode == "Webcam":
        st.write("Real-time webcam feed with ASL recognition.")
        
        # Check if webcam is available (not in cloud environment)
        test_cap = cv2.VideoCapture(0)
        if not test_cap.isOpened():
            st.info("Webcam mode is only available when running locally. For cloud deployment, please use Image Upload mode.")
            st.write("To use webcam mode locally, run this app with: `streamlit run streamlit_app.py`")
        else:
            test_cap.release()
            start_btn = st.button("Start Webcam")
            stop_btn = st.button("Stop Webcam")
            
            FRAME_WINDOW = st.image([])
            
            if start_btn:
                from src.realtime import run_realtime
                # Use generator mode to get frames
                frame_gen = run_realtime("models/asl_model.pth", use_gui=False)
                
                for frame in frame_gen:
                    if stop_btn:
                        break
                    # Convert BGR to RGB for streamlit
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    FRAME_WINDOW.image(frame_rgb)

st.sidebar.markdown("---")
st.sidebar.subheader("Prediction History")
st.sidebar.write(" ".join(st.session_state.history[-20:]))
