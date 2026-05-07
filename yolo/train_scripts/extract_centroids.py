import cv2
import os
import sys
import numpy as np
import pandas as pd
import glob

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_ROOT = os.path.dirname(CURRENT_DIR)
REALVVA_ROOT = os.path.dirname(YOLO_ROOT)
sys.path.append(REALVVA_ROOT)
import project_paths  # noqa: F401

from ultralytics import YOLO

# --- CONFIG ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_ROOT = os.path.dirname(CURRENT_DIR)
REALVVA_ROOT = os.path.dirname(YOLO_ROOT)
DATA_ROOT = os.path.join(REALVVA_ROOT, "data")

MODEL_PATH = os.path.join(YOLO_ROOT, "models", "run", "weights", "best.pt")
OUTPUT_NPZ = os.path.join(YOLO_ROOT, "data", "policy_dataset.npz")

# Classes: 0-5 Joints, 6 Target
NUM_CLASSES = 7 

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print(f"Loading Model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    csv_files = glob.glob(os.path.join(DATA_ROOT, "jointdata", "*.csv"))
    print(f"Found {len(csv_files)} joint recording sessions.")
    
    all_X = [] 
    all_Y = [] 
    
    for csv_path in csv_files:
        basename = os.path.basename(csv_path)
        # jd1.csv -> 1
        idx_str = basename.replace("jd", "").replace(".csv", "")
        if not idx_str.isdigit():
             # Handle potential session_name timestamp case if strict naming not followed
             # Just try finding matching video by name similarity if strictly standard
             pass

        # Try finding video
        vid_path_demo = os.path.join(DATA_ROOT, "demovideos", f"demovid{idx_str}.mp4")
        vid_path_cam = os.path.join(DATA_ROOT, "camview", f"camview{idx_str}.mp4")
        
        # Also try matching timestamp name: rec_1234_joints.csv -> camview/rec_1234.mp4
        vid_path_rec = os.path.join(DATA_ROOT, "camview", basename.replace("_joints.csv", ".mp4"))

        if os.path.exists(vid_path_demo): vid_path = vid_path_demo
        elif os.path.exists(vid_path_cam): vid_path = vid_path_cam
        elif os.path.exists(vid_path_rec): vid_path = vid_path_rec
        else:
            print(f"Skipping {basename}: No matching video found.")
            continue
            
        print(f"Processing: {basename}")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        # Columns: timestamp,j0,j1,j2,j3,j4,j5...
        actions = df[['j0','j1','j2','j3','j4','j5']].values 
        
        cap = cv2.VideoCapture(vid_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        limit = min(total_frames, len(actions))
        print(f"  Frames: {limit}")

        for i in range(limit):
            ret, frame = cap.read()
            if not ret: break
            
            # Low threshold to catch everything
            results = model(frame, verbose=False, conf=0.05)
            
            # Feature Vector: 7 classes * 2 coords (x,y) = 14 dims
            features = np.zeros(14)
            
            # Track which classes found to filter bad frames
            found_classes = [False] * 7
            
            if results[0].boxes:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    if cls_id < 7:
                        # If multiple boxes for same class, take first (highest conf)
                        if not found_classes[cls_id]:
                             x, y, w, h = box.xywh[0].cpu().numpy()
                             
                             # Normalize
                             H, W = frame.shape[:2]
                             nx = x / W
                             ny = y / H
                             
                             features[cls_id*2] = nx
                             features[cls_id*2+1] = ny
                             found_classes[cls_id] = True
            
            # Simple integrity check: 
            # If we miss ANY joint (0-5), the data is useless for training
            # (Target might be missing if out of frame, but let's be strict for now)
            # Actually, standard approach is:
            # If missing, we could interpolate, but simpler to just DROP specific frame
            # if we have enough data. 
            
            if all(found_classes[0:6]): # J0-J5 available
                all_X.append(features)
                all_Y.append(actions[i])
                
        cap.release()
        
    X_data = np.array(all_X)
    Y_data = np.array(all_Y)
    
    print(f"Extraction Complete.")
    print(f"Valid Samples: {len(X_data)}")
    
    if len(X_data) > 0:
        np.savez(OUTPUT_NPZ, X=X_data, Y=Y_data)
        print(f"Saved dataset to: {OUTPUT_NPZ}")
    else:
        print("Error: No valid data extracted! Check thresholds or labels.")

if __name__ == "__main__":
    main()
