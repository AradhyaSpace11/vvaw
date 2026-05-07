import os
import sys
import cv2
import torch
import numpy as np
import warnings
import pandas as pd
from PIL import Image
from project_paths import CAMVIEW_DIR, DATASET_3D, DEMOVIDEO_DIR, RAWDATA_JOINT_DIR, YOLO_DIR, YOLO_WEIGHTS
from transformers import pipeline
from ultralytics import YOLO

# Suppress HuggingFace sequential pipeline warning
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.pipelines")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

sys.path.append(str(YOLO_DIR))
from utils.smoothing import CentroidSmoother

def process_video(vid_path, yolo_model, depth_estimator, limit=None):
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print(f"Error opening video {vid_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if limit is not None:
        limit = min(total_frames, limit)
    else:
        limit = total_frames

    last_known_centroids = {}
    xy_smoother = CentroidSmoother()
    depth_ema = {}
    EMA_ALPHA = 0.6  
    
    gripper_min_dist = float('inf')
    gripper_max_dist = float('-inf')
    
    depth_min = float('inf')
    depth_max = float('-inf')

    precomputed_data = []

    print(f"\nProcessing {os.path.basename(vid_path)}...")
    frame_idx = 0
    
    while cap.isOpened():
        if limit is not None and frame_idx >= limit:
            break
            
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"\r  Frame {frame_idx}/{limit}", end="", flush=True)

        h_orig, w_orig = frame.shape[:2]

        yolo_results = yolo_model(frame, verbose=False, conf=0.065)
        current_centroids = {}
        found_in_frame = [False] * 7

        if yolo_results[0].boxes:
            for box in yolo_results[0].boxes:
                ids = int(box.cls[0].item())
                x, y, w, h = box.xywh[0].cpu().numpy()
                cx, cy = int(x), int(y)
                
                if ids in last_known_centroids:
                    lx, ly = last_known_centroids[ids]
                    dist = ((cx - lx)**2 + (cy - ly)**2)**0.5
                    if dist > 40:
                        continue
                current_centroids[ids] = (cx, cy)
                found_in_frame[ids] = True

        if 4 in current_centroids and 5 not in current_centroids:
            if 4 in last_known_centroids and 5 in last_known_centroids:
                dx = current_centroids[4][0] - last_known_centroids[4][0]
                dy = current_centroids[4][1] - last_known_centroids[4][1]
                current_centroids[5] = (last_known_centroids[5][0] + dx, last_known_centroids[5][1] + dy)
                found_in_frame[5] = True
            else:
                current_centroids[5] = current_centroids[4]
                
        elif 5 in current_centroids and 4 not in current_centroids:
            if 4 in last_known_centroids and 5 in last_known_centroids:
                dx = current_centroids[5][0] - last_known_centroids[5][0]
                dy = current_centroids[5][1] - last_known_centroids[5][1]
                current_centroids[4] = (last_known_centroids[4][0] + dx, last_known_centroids[4][1] + dy)
                found_in_frame[4] = True
            else:
                current_centroids[4] = current_centroids[5]

        real_detections = set(current_centroids.keys())

        # Persistence for Pass 1 calculation
        for i in range(7):
            if i not in current_centroids and i in last_known_centroids:
                current_centroids[i] = last_known_centroids[i]

        for ids, pos in current_centroids.items():
            last_known_centroids[ids] = pos

        smoothed_centroids = {}
        for ids, (cx, cy) in current_centroids.items():
            sx, sy = xy_smoother.update(ids, cx, cy)
            smoothed_centroids[ids] = (int(sx), int(sy))

        new_w = 512
        new_h = int(h_orig * (new_w / w_orig))
        small_frame = cv2.resize(frame, (new_w, new_h))
        frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        with torch.no_grad():
            depth_result = depth_estimator(pil_img)
        
        depth_np = np.array(depth_result["depth"]).astype(np.float32)

        ground_profile = np.median(depth_np, axis=1)
        depth_np = depth_np - ground_profile[:, np.newaxis]

        joint_depths = {}
        for ids, (cx, cy) in smoothed_centroids.items():
            if ids in real_detections:
                dx_coord = int(cx * (new_w / w_orig))
                dy_coord = int(cy * (new_h / h_orig))
                dx_coord = max(0, min(new_w - 1, dx_coord))
                dy_coord = max(0, min(new_h - 1, dy_coord))
                
                raw_depth = depth_np[dy_coord, dx_coord]
                
                if ids not in depth_ema:
                    depth_ema[ids] = raw_depth
                else:
                    depth_ema[ids] = EMA_ALPHA * raw_depth + (1 - EMA_ALPHA) * depth_ema[ids]
                
            smoothed_depth = depth_ema.get(ids, 0.0)
            joint_depths[ids] = float(smoothed_depth)
            
            if smoothed_depth < depth_min: depth_min = smoothed_depth
            if smoothed_depth > depth_max: depth_max = smoothed_depth
            
        gripper_dist = 0.0
        if 4 in smoothed_centroids and 5 in smoothed_centroids:
            x4, y4 = smoothed_centroids[4]
            x5, y5 = smoothed_centroids[5]
            gripper_dist = ((x4 - x5)**2 + (y4 - y5)**2)**0.5
            if gripper_dist < gripper_min_dist: gripper_min_dist = gripper_dist
            if gripper_dist > gripper_max_dist: gripper_max_dist = gripper_dist

        precomputed_data.append({
            'centroids': smoothed_centroids,
            'depths': joint_depths,
            'gripper_dist': gripper_dist,
            'found_in_frame': found_in_frame,
            'w_orig': w_orig,
            'h_orig': h_orig
        })
        
    cap.release()
    print(f"\n  Done inference. Building matrix...")

    video_X = []
    
    if len(precomputed_data) == 0:
        return np.array([])
        
    current_features = np.zeros(22)
    
    # 1. Backfill Pass: Find the first valid observation for each feature
    found_initial = [False] * 7
    found_initial_gripper = False
    
    for data in precomputed_data:
        centroids = data['centroids']
        depths = data['depths']
        w_orig = data['w_orig']
        h_orig = data['h_orig']
        
        for ids in range(7):
            if ids in centroids and not found_initial[ids]:
                nx = centroids[ids][0] / w_orig
                ny = centroids[ids][1] / h_orig
                nz = 0.0
                if depth_max > depth_min:
                    nz = (depths.get(ids, 0.0) - depth_min) / (depth_max - depth_min)
                
                current_features[ids*3] = nx
                current_features[ids*3 + 1] = ny
                current_features[ids*3 + 2] = nz
                found_initial[ids] = True
                
        if 4 in centroids and 5 in centroids and not found_initial_gripper:
            g_dist = data['gripper_dist']
            intensity = 0.5
            if gripper_max_dist > gripper_min_dist:
                intensity = (g_dist - gripper_min_dist) / (gripper_max_dist - gripper_min_dist)
            current_features[21] = intensity
            found_initial_gripper = True
            
        if all(found_initial) and found_initial_gripper:
            break

    # 2. Main Pass: Process all frames
    for data in precomputed_data:
        centroids = data['centroids']
        depths = data['depths']
        w_orig = data['w_orig']
        h_orig = data['h_orig']
        
        # Update current features for items found or inferred in this frame
        for ids in range(7):
            if ids in centroids:
                nx = centroids[ids][0] / w_orig
                ny = centroids[ids][1] / h_orig
                nz = 0.0
                if depth_max > depth_min:
                    nz = (depths.get(ids, 0.0) - depth_min) / (depth_max - depth_min)
                
                current_features[ids*3] = nx
                current_features[ids*3 + 1] = ny
                current_features[ids*3 + 2] = nz

        # Update Gripper intensity if possible
        if 4 in centroids and 5 in centroids:
            g_dist = data['gripper_dist']
            intensity = 0.5
            if gripper_max_dist > gripper_min_dist:
                intensity = (g_dist - gripper_min_dist) / (gripper_max_dist - gripper_min_dist)
            current_features[21] = intensity
        
        video_X.append(current_features.copy())
            
    return np.array(video_X)


def main():
    YOLO_MODEL_PATH = str(YOLO_WEIGHTS)
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"Error: Model not found at {YOLO_MODEL_PATH}")
        return

    print("Loading YOLO model...")
    yolo_model = YOLO(YOLO_MODEL_PATH)

    device = 0 if torch.cuda.is_available() else -1
    print(f"Using Device: {'GPU' if device == 0 else 'CPU'} for Depth Model")
    print("Loading Depth Anything model...")
    depth_estimator = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf", device=device)

    camview_dir = str(CAMVIEW_DIR)
    demovideo_dir = str(DEMOVIDEO_DIR)
    jointdata_dir = str(RAWDATA_JOINT_DIR)
    
    output_npy_path = str(DATASET_3D)
    
    dataset_list = []
    
    # We will search from index 1 to 999
    # A valid trial must have demovid{i}.mp4, camview{i}.mp4, and jd{i}.csv
    
    for i in range(1, 1000):
        demo_path = os.path.join(demovideo_dir, f"demovid{i}.mp4")
        cam_path = os.path.join(camview_dir, f"camview{i}.mp4")
        jd_path = os.path.join(jointdata_dir, f"jd{i}.csv")
        
        if os.path.exists(demo_path) and os.path.exists(cam_path) and os.path.exists(jd_path):
            print(f"\n=====================================")
            print(f"   STARTING TRIAL {i}")
            print(f"=====================================")
            
            # Load Actions
            try:
                df = pd.read_csv(jd_path)
                actions = df[['j0','j1','j2','j3','j4','j5']].values 
            except Exception as e:
                print(f"Error loading {jd_path}: {e}")
                continue
                
            print(f"--> Processing Demovideo {i}")
            demo_X = process_video(demo_path, yolo_model, depth_estimator, limit=None)
            
            if demo_X is None or len(demo_X) == 0:
                print(f"Failed to process demovideo {i}")
                continue
                
            print(f"\n--> Processing Camview {i}")
            cam_X = process_video(cam_path, yolo_model, depth_estimator, limit=len(actions))
            
            if cam_X is None or len(cam_X) == 0:
                print(f"Failed to process camview {i}")
                continue
                
            # Align actions strictly with the output of cam_X in case the physical video ended slightly early
            actual_len = len(cam_X)
            aligned_actions = actions[:actual_len]
            
            trial_data = {
                'trial': i,
                'demo_X': demo_X,
                'cam_X': cam_X,
                'actions': aligned_actions
            }
            dataset_list.append(trial_data)
            print(f"Trial {i} saved successfully! (Demo frames: {len(demo_X)}, Cam frames: {len(cam_X)})")

    print("\nExtraction and Preprocessing Complete.")
    print(f"Valid Trials Processed: {len(dataset_list)}")
    
    if len(dataset_list) > 0:
        np.save(output_npy_path, np.array(dataset_list, dtype=object))
        print(f"Saved highly structured ready-to-train dataset to: {output_npy_path}")

if __name__ == "__main__":
    main()
