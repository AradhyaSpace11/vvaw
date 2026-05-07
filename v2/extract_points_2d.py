import os
import sys
import cv2
import numpy as np
import pandas as pd

# Set up ROOT_DIR
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VVA_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(VVA_DIR)
from project_paths import CAMVIEW_DIR, DEMOVIDEO_DIR, RAWDATA_JOINT_DIR, YOLO_DIR, YOLO_WEIGHTS
from ultralytics import YOLO
sys.path.append(str(YOLO_DIR))
from utils.smoothing import CentroidSmoother

def process_video_2d(vid_path, yolo_model, limit=None):
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print(f"Error opening video {vid_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    if limit is not None: limit = min(total_frames, limit)
    else: limit = total_frames

    last_known_centroids = {}
    xy_smoother = CentroidSmoother()
    precomputed_data = []

    print(f"\nProcessing {os.path.basename(vid_path)}...")
    frame_idx = 0
    
    while cap.isOpened():
        if limit is not None and frame_idx >= limit: break
            
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        if frame_idx % 10 == 0: print(f"\r  Frame {frame_idx}/{limit}", end="", flush=True)

        yolo_results = yolo_model(frame, verbose=False, conf=0.065)
        current_centroids = {}

        if yolo_results[0].boxes:
            for box in yolo_results[0].boxes:
                ids = int(box.cls[0].item())
                x, y, w, h = box.xywh[0].cpu().numpy()
                cx, cy = int(x), int(y)
                
                if ids in last_known_centroids:
                    lx, ly = last_known_centroids[ids]
                    dist = ((cx - lx)**2 + (cy - ly)**2)**0.5
                    if dist > 40: continue
                current_centroids[ids] = (cx, cy)

        if 4 in current_centroids and 5 not in current_centroids:
            if 4 in last_known_centroids and 5 in last_known_centroids:
                dx = current_centroids[4][0] - last_known_centroids[4][0]
                dy = current_centroids[4][1] - last_known_centroids[4][1]
                current_centroids[5] = (last_known_centroids[5][0] + dx, last_known_centroids[5][1] + dy)
            else: current_centroids[5] = current_centroids[4]
                
        elif 5 in current_centroids and 4 not in current_centroids:
            if 4 in last_known_centroids and 5 in last_known_centroids:
                dx = current_centroids[5][0] - last_known_centroids[5][0]
                dy = current_centroids[5][1] - last_known_centroids[5][1]
                current_centroids[4] = (last_known_centroids[4][0] + dx, last_known_centroids[4][1] + dy)
            else: current_centroids[4] = current_centroids[5]

        for i in range(7):
            if i not in current_centroids and i in last_known_centroids:
                current_centroids[i] = last_known_centroids[i]
        for ids, pos in current_centroids.items():
            last_known_centroids[ids] = pos

        smoothed_centroids = {}
        for ids, (cx, cy) in current_centroids.items():
            sx, sy = xy_smoother.update(ids, cx, cy)
            smoothed_centroids[ids] = (sx, sy)

        precomputed_data.append({'centroids': smoothed_centroids})
        
    cap.release()
    print(f"\n  Done inference. Building 2D Feature matrix...")

    if not precomputed_data: return np.array([])
        
    # 1. Backfill Pass to find initial states
    persistent_centroids = {i: None for i in range(7)}
    for data in precomputed_data:
        for i in range(7):
            if persistent_centroids[i] is None and i in data['centroids']:
                persistent_centroids[i] = data['centroids'][i]
        if all(v is not None for v in persistent_centroids.values()):
            break

    video_X = []
    
    # 2. Extract Absolute Coordinates (14 Dimensions)
    for data in precomputed_data:
        # Update state
        for i in range(7):
            if i in data['centroids']:
                persistent_centroids[i] = data['centroids'][i]
                
        features = np.zeros(14)
        for i in range(7):
            cx, cy = persistent_centroids[i]
            features[i*2] = cx / W
            features[i*2 + 1] = cy / H
            
        video_X.append(features.copy())
            
    return np.array(video_X)


def main():
    YOLO_MODEL_PATH = str(YOLO_WEIGHTS)
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"Error: Model not found at {YOLO_MODEL_PATH}")
        return

    print("Loading YOLO model...")
    yolo_model = YOLO(YOLO_MODEL_PATH)

    camview_dir = str(CAMVIEW_DIR)
    demovideo_dir = str(DEMOVIDEO_DIR)
    jointdata_dir = str(RAWDATA_JOINT_DIR)
    
    output_npy_path = os.path.join(CURRENT_DIR, "dataset_2d.npy")
    dataset_list = []
    
    for i in range(1, 1000):
        demo_path = os.path.join(demovideo_dir, f"demovid{i}.mp4")
        cam_path = os.path.join(camview_dir, f"camview{i}.mp4")
        jd_path = os.path.join(jointdata_dir, f"jd{i}.csv")
        
        if os.path.exists(demo_path) and os.path.exists(cam_path) and os.path.exists(jd_path):
            print(f"\n=====================================")
            print(f"   STARTING TRIAL {i}")
            print(f"=====================================")
            
            try:
                df = pd.read_csv(jd_path)
                actions = df[['j0','j1','j2','j3','j4','j5']].values 
            except Exception as e:
                print(f"Error loading {jd_path}: {e}")
                continue
                
            print(f"--> Processing Demovideo {i}")
            demo_X = process_video_2d(demo_path, yolo_model, limit=None)
            if demo_X is None or len(demo_X) == 0: continue
                
            print(f"\n--> Processing Camview {i}")
            cam_X = process_video_2d(cam_path, yolo_model, limit=len(actions))
            if cam_X is None or len(cam_X) == 0: continue
                
            actual_len = len(cam_X)
            aligned_actions = actions[:actual_len]
            
            trial_data = {
                'trial': i,
                'demo_X': demo_X,
                'cam_X': cam_X,
                'actions': aligned_actions
            }
            dataset_list.append(trial_data)
            print(f"Trial {i} saved successfully! (Demo: {len(demo_X)}, Cam: {len(cam_X)})")

    print("\nExtraction and Preprocessing Complete.")
    print(f"Valid Trials: {len(dataset_list)}")
    
    if len(dataset_list) > 0:
        np.save(output_npy_path, np.array(dataset_list, dtype=object))
        print(f"Saved highly structured ready-to-train dataset to: {output_npy_path}")

if __name__ == "__main__":
    main()
