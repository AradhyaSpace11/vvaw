import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# --- CONFIG ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VVA_DIR = os.path.dirname(CURRENT_DIR)
DATA_NPY = os.path.join(VVA_DIR, "dataset.npy")

MODEL_OUT = os.path.join(CURRENT_DIR, "model1_vla.pth")

# --- DATASET ---
class VLADataset(Dataset):
    def __init__(self, data_path):
        print(f"Loading Dataset: {data_path}")
        raw_data = np.load(data_path, allow_pickle=True)
        
        self.samples = []
        
        # Find maximum demovideo length across all trials to pad them equally
        max_demo_len = max([len(t['demo_X']) for t in raw_data])
        print(f"Max Demovideo Sequence Length: {max_demo_len}")
        
        for trial in raw_data:
            demo = trial['demo_X']
            cam = trial['cam_X']
            actions = trial['actions']
            
            # Pad the demovideo sequence with zeros up to max_demo_len
            pad_len = max_demo_len - len(demo)
            demo_padded = np.pad(demo, ((0, pad_len), (0, 0)), mode='constant')
            
            # Create a boolean mask where True means "this is padding, ignore it"
            demo_mask = np.zeros(max_demo_len, dtype=bool)
            demo_mask[len(demo):] = True
            
            # Generate a training sample for EVERY frame of the camview
            # Target = Action at time T
            for t in range(len(cam)):
                self.samples.append({
                    'demo_seq': torch.tensor(demo_padded, dtype=torch.float32),
                    'demo_mask': torch.tensor(demo_mask, dtype=torch.bool),
                    'cam_state': torch.tensor(cam[t], dtype=torch.float32),
                    'action': torch.tensor(actions[t], dtype=torch.float32)
                })
                
        print(f"Created {len(self.samples)} total training samples across {len(raw_data)} trials.")
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]

# --- VLA ARCHITECTURE ---
class VLAPromptPolicy(nn.Module):
    def __init__(self, input_dim=22, embed_dim=128, action_dim=6):
        super().__init__()
        
        # 1. Embed Demo Sequence
        # Uses LSTM to read the sequence of 22-feature frames and output tokens
        self.demo_embed = nn.LSTM(input_dim, embed_dim, batch_first=True)
        
        # 2. Embed Camview State (Current Frame)
        self.state_embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 3. Cross Attention
        # The Camview state queries the entire Demovideo sequence to find relevant context
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        
        # 4. Action Head
        # Takes the state AND the attention context, outputs the joint angles
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, demo_seq, demo_mask, cam_state):
        # Encode demo sequence -> tokens [Batch, MaxDemoLen, EmbedDim]
        demo_tokens, _ = self.demo_embed(demo_seq)
        
        # Encode current camview state -> [Batch, EmbedDim]
        state_query = self.state_embed(cam_state)
        # Reshape to [Batch, 1, EmbedDim] because Attention expects sequence length dimension
        state_query = state_query.unsqueeze(1)
        
        # Cross Attention Execution
        # We pass demo_mask as key_padding_mask so it completely ignores the zero-padding
        attn_out, _ = self.attention(
            query=state_query, 
            key=demo_tokens, 
            value=demo_tokens, 
            key_padding_mask=demo_mask
        )
        
        # attn_out is [Batch, 1, EmbedDim]
        attn_out = attn_out.squeeze(1) # Back to [Batch, EmbedDim]
        state_features = state_query.squeeze(1)
        
        # Merge State + Relevant Demo Context
        combined = torch.cat([state_features, attn_out], dim=-1) # [Batch, EmbedDim * 2]
        
        # Predict 6 Joint Actions
        action = self.action_head(combined)
        return action

# --- TRAINING SCRIPT ---
def train():
    if not os.path.exists(DATA_NPY):
        print(f"Error: Dataset not found at {DATA_NPY}")
        print("Please run extract_points.py first to generate the .npy dataset.")
        return

    # Load Data
    dataset = VLADataset(DATA_NPY)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Setup Device (Optimized for RTX 3050)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training VLA Policy on: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    model = VLAPromptPolicy().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    epochs = 500
    
    print(f"Starting Training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch in dataloader:
            # Move to GPU
            demo_seq = batch['demo_seq'].to(device)
            demo_mask = batch['demo_mask'].to(device)
            cam_state = batch['cam_state'].to(device)
            target_action = batch['action'].to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass
            pred_action = model(demo_seq, demo_mask, cam_state)
            
            # Loss & Backprop
            loss = criterion(pred_action, target_action)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Avg Loss: {avg_loss:.6f}")
            
    # Save Final Model
    torch.save(model.state_dict(), MODEL_OUT)
    print(f"\n=========================================")
    print(f"Training Complete! Model saved successfully:")
    print(f"-> {MODEL_OUT}")
    print(f"=========================================")

if __name__ == "__main__":
    train()
