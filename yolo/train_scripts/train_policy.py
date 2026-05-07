import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys

# --- CONFIG ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_ROOT = os.path.dirname(CURRENT_DIR)
DATA_NPZ = os.path.join(YOLO_ROOT, "data", "policy_dataset.npz")
MODEL_OUT = os.path.join(YOLO_ROOT, "models", "policy_mlp_detect.pth")

# --- MODEL (14 -> 6) ---
class RobotPolicy(nn.Module):
    def __init__(self, input_dim=14, output_dim=6):
        super(RobotPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

def train():
    if not os.path.exists(DATA_NPZ):
        print(f"Error: Dataset not found at {DATA_NPZ}")
        return

    print(f"Loading Dataset: {DATA_NPZ}")
    data = np.load(DATA_NPZ)
    X_raw = data['X'] # (N, 14)
    Y_raw = data['Y'] # (N, 6)
    
    X_tensor = torch.tensor(X_raw, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_raw, dtype=torch.float32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    model = RobotPolicy().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    epochs = 1000
    batch_size = 32
    num_samples = len(X_raw)
    
    X_tensor = X_tensor.to(device)
    Y_tensor = Y_tensor.to(device)
    
    print(f"Starting Training on {num_samples} samples...")
    for epoch in range(epochs):
        indices = torch.randperm(num_samples)
        epoch_loss = 0.0
        batches = 0
        
        for i in range(0, num_samples, batch_size):
            idxs = indices[i:i+batch_size]
            x_batch = X_tensor[idxs]
            y_batch = Y_tensor[idxs]
            
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batches += 1
            
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/batches:.6f}")
            
    torch.save(model.state_dict(), MODEL_OUT)
    print(f"Policy Saved: {MODEL_OUT}")

if __name__ == "__main__":
    train()
