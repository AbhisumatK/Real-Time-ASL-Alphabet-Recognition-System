# Real-Time ASL Alphabet Recognition System

This project implements a real-time American Sign Language (ASL) alphabet recognition system using MediaPipe for hand landmark extraction and a PyTorch Deep Neural Network for classification.

## Project Structure

- `data/`: Dataset storage (ignored by git).
- `models/`: Saved models and evaluation plots.
- `notebooks/`: Data exploration and analysis.
- `src/`: Source code.
    - `dataset.py`: Data loading and landmark extraction.
    - `model.py`: Neural Network architecture.
    - `train.py`: Training script.
    - `evaluate.py`: Evaluation script.
    - `predict.py`: Single image inference.
    - `realtime.py`: Webcam-based real-time recognition.
    - `utils.py`: Utility functions.
    - `baseline.py`: Baseline Random Forest model.
- `streamlit_app.py`: Web application interface.

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Download Dataset
The project uses `kagglehub` to download the ASL Alphabet dataset. Run:
```bash
python src/utils.py
```

### 2. Training
Train the neural network model:
```bash
python src/train.py
```

### 3. Evaluation
Evaluate the trained model:
```bash
python src/evaluate.py
```

### 4. Real-Time Recognition
Run the webcam-based recognition:
```bash
python src/realtime.py
```

### 5. Streamlit App
Launch the web interface:
```bash
streamlit run streamlit_app.py
```

## Why Hand Landmarks?
Instead of classifying raw images, we extract 21 hand landmarks (63 features). This approach is:
- **Lightweight**: The model is much smaller and faster.
- **Robust to Lighting/Background**: Landmarks focus only on hand geometry.
- **Scalable**: Works well on low-power devices.
