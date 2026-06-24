import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from src.dataset import ASLDataset, get_dataloader
from src.model import get_model
from src.utils import get_asl_labels

def train_model(
    data_path: str,
    npz_path: str = "data/asl_landmarks.npz",
    model_save_path: str = "models/asl_model.pth",
    batch_size: int = 64,
    epochs: int = 50,
    lr: float = 0.001,
    patience: int = 5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = get_asl_labels()
    num_classes = len(labels)
    
    # Dataset setup
    if os.path.exists(npz_path):
        print(f"Loading pre-extracted landmarks from {npz_path}...")
        from src.landmark_dataset import LandmarkDataset
        full_dataset = LandmarkDataset(npz_path)
    else:
        print(f"Pre-extracted landmarks not found at {npz_path}. Using slow image-based loading...")
        full_dataset = ASLDataset(data_path, labels)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = get_model(num_classes=num_classes, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({'loss': running_loss/len(train_loader), 'acc': 100.*correct/total})
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_loss)
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            epochs_no_improve = 0
            print(f"Saved best model to {model_save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered!")
                break
                
    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('Loss Curves')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.title('Accuracy Curves')
    plt.legend()
    
    plt.savefig('models/learning_curves.png')
    plt.show()

if __name__ == "__main__":
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Path provided by user
    DATA_PATH = r"C:/Users/ABHISUMAT/.cache/kagglehub/datasets/grassknoted/asl-alphabet/versions/1/asl_alphabet_train/asl_alphabet_train"
    
    train_model(
        data_path=DATA_PATH,
        model_save_path="models/asl_model.pth",
        batch_size=64,
        epochs=30, # Reduced for initial run
        lr=0.001,
        patience=5
    )
