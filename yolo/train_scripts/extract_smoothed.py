import cv2
import os
import sys
import numpy as np
import pandas as pd
import glob
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_ROOT = os.path.dirname(CURRENT_DIR)
REALVVA_ROOT = os.path.dirname(YOLO_ROOT)
sys.path.append(REALVVA_ROOT)
import project_paths  # noqa: F401

from ultralytics import YOLO

# Helper Import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.smoothing import CentroidSmoother

# --- CONFIG ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_ROOT = os.path.dirname(CURRENT_DIR)
REALVVA_ROOT = os.path.dirname(YOLO_ROOT)
DATA_ROOT = os.path.join(REALVVA_ROOT, "data")

MODEL_PATH = os.path.join(YOLO_ROOT, "models", "run", "weights", "best.pt")
OUTPUT_NPZ = os.path.join(YOLO_ROOT, "data", "policy_dataset_smoothed.npz")

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print(f"Loading Model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    csv_files = glob.glob(os.path.join(DATA_ROOT, "jointdata", "*.csv"))
    print(f"Found {len(csv_files)} sessions.")
    
    all_X = [] 
    all_Y = [] 
    
    for csv_path in csv_files:
        basename = os.path.basename(csv_path)
        print(f"Processing: {basename}")
        
        # Match Video
        vid_path_rec = os.path.join(DATA_ROOT, "camview", basename.replace("_joints.csv", ".mp4"))
        # Fallback logic if needed (simple for now)
        vid_path = vid_path_rec
        
        if not os.path.exists(vid_path):
             # Try demo naming convention
             idx_str = basename.replace("jd", "").replace(".csv", "")
             vid_path = os.path.join(DATA_ROOT, "demovideos", f"demovid{idx_str}.mp4")
             
        if not os.path.exists(vid_path):
            print("  Video not found, skipping.")
            continue

        # Load Data
        df = pd.read_csv(csv_path)
        actions = df[['j0','j1','j2','j3','j4','j5']].values 
        
        cap = cv2.VideoCapture(vid_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        limit = min(total_frames, len(actions))
        
        # Init Smoother for THIS video
        smoother = CentroidSmoother()
        
        # PERSISTENCE: Keep last valid state (Zero-Order Hold)
        current_features = np.zeros(14)
        # We also need to know if we have ever found the joints to start recording
        has_initialized = False
        
        for i in range(limit):
            ret, frame = cap.read()
            if not ret: break
            
            # Predict
            results = model(frame, verbose=False, conf=0.05)
            
            # Track what we found in THIS frame
            found_in_frame = [False] * 7
            
            if results[0].boxes:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    if cls_id < 7:
                         if not found_in_frame[cls_id]:
                             x, y, w, h = box.xywh[0].cpu().numpy()
                             
                             # SMOOTHING
                             sx, sy = smoother.update(cls_id, x, y)
                             
                             # Normalize
                             H, W = frame.shape[:2]
                             nx = sx / W
                             ny = sy / H
                             
                             # Update Persistent State
                             current_features[cls_id*2] = nx
                             current_features[cls_id*2+1] = ny
                             found_in_frame[cls_id] = True
            
            # Check initialization (Need J0-J5 at least once to start)
            if not has_initialized:
                if all(found_in_frame[0:6]):
                    has_initialized = True
            
            # If initialized, we save every frame using the persistent state
            # (Features carry over from previous frame if missing in this one)
            if has_initialized:
                all_X.append(current_features.copy())
                all_Y.append(actions[i])
                
        cap.release()
        
    X_data = np.array(all_X)
    Y_data = np.array(all_Y)
    
    print(f"Extraction (Smoothed) Complete.")
    print(f"Valid Samples: {len(X_data)}")
    
    if len(X_data) > 0:
        np.savez(OUTPUT_NPZ, X=X_data, Y=Y_data)
        print(f"Saved dataset to: {OUTPUT_NPZ}")

if __name__ == "__main__":
    main()
