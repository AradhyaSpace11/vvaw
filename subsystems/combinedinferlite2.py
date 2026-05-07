import os
import sys
import cv2
import torch
import numpy as np
import warnings
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_paths import CAMVIEW_DIR, YOLO_DIR, YOLO_WEIGHTS
from transformers import pipeline
from ultralytics import YOLO

# Suppress HuggingFace sequential pipeline warning
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.pipelines")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Add local YOLO utilities to path for XY smoothing
sys.path.append(str(YOLO_DIR))
from utils.smoothing import CentroidSmoother

def get_color_from_depth(val, min_val, max_val):
    """Maps a normalized depth value to an INFERNO colormap color."""
    if max_val <= min_val or min_val == float('inf'):
        norm = 128
    else:
        norm = int(255 * (val - min_val) / (max_val - min_val))
    
    norm = max(0, min(255, norm))
    pixel = np.array([[norm]], dtype=np.uint8)
    color = cv2.applyColorMap(pixel, cv2.COLORMAP_INFERNO)[0][0]
    return (int(color[0]), int(color[1]), int(color[2]))

def main():
    # Setup paths from the standalone VVA project root
    YOLO_MODEL_PATH = str(YOLO_WEIGHTS)
    VIDEO_DIR = str(CAMVIEW_DIR)

    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"Error: Model not found at {YOLO_MODEL_PATH}")
        return

    # Ask for video number
    vid_num = input("Enter camview video number (e.g., 1 for camview1.mp4): ").strip()
    if not vid_num:
        print("No number entered. Exiting.")
        return
        
    vid_name = f"camview{vid_num}.mp4"
    vid_path = os.path.join(VIDEO_DIR, vid_name)
    
    if not os.path.exists(vid_path):
        print(f"Error: Video not found at {vid_path}")
        return

    # 1. Load Models
    print(f"Loading YOLO model from {YOLO_MODEL_PATH}...")
    yolo_model = YOLO(YOLO_MODEL_PATH)

    device = 0 if torch.cuda.is_available() else -1
    print(f"Using Device: {'GPU (CUDA)' if device == 0 else 'CPU'} for Depth Model")
    print("Loading Depth Anything model...")
    try:
        depth_estimator = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf", device=device)
    except Exception as e:
        print(f"Failed to load depth model: {e}")
        return

    cap = cv2.VideoCapture(vid_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # State for YOLO (XY)
    last_known_centroids = {}
    xy_smoother = CentroidSmoother()
    
    # State for Depth Smoothing (Exponential Moving Average)
    depth_ema = {}
    EMA_ALPHA = 0.6  
    running_depth_min = float('inf')
    running_depth_max = float('-inf')
    
    gripper_min_dist = float('inf')
    gripper_max_dist = float('-inf')

    # Data array to hold all precomputed coordinates and colors (very low memory)
    precomputed_data = []

    print(f"\n--- PASS 1: Precomputing Inference Data ---")
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        print(f"\rProcessing frame {frame_idx}/{total_frames}...", end="", flush=True)

        h_orig, w_orig = frame.shape[:2]

        # ---------------------------------------------------------
        # A. YOLO INFERENCE
        # ---------------------------------------------------------
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
                    if dist > 40:
                        continue
                current_centroids[ids] = (cx, cy)

        if 4 in current_centroids and 5 not in current_centroids:
            if 4 in last_known_centroids and 5 in last_known_centroids:
                dx = current_centroids[4][0] - last_known_centroids[4][0]
                dy = current_centroids[4][1] - last_known_centroids[4][1]
                current_centroids[5] = (last_known_centroids[5][0] + dx, last_known_centroids[5][1] + dy)
            else:
                current_centroids[5] = current_centroids[4]
                
        elif 5 in current_centroids and 4 not in current_centroids:
            if 4 in last_known_centroids and 5 in last_known_centroids:
                dx = current_centroids[5][0] - last_known_centroids[5][0]
                dy = current_centroids[5][1] - last_known_centroids[5][1]
                current_centroids[4] = (last_known_centroids[4][0] + dx, last_known_centroids[4][1] + dy)
            else:
                current_centroids[4] = current_centroids[5]

        real_detections = set(current_centroids.keys())

        for i in range(7):
            if i not in current_centroids and i in last_known_centroids:
                current_centroids[i] = last_known_centroids[i]

        for ids, pos in current_centroids.items():
            last_known_centroids[ids] = pos

        smoothed_centroids = {}
        for ids, (cx, cy) in current_centroids.items():
            sx, sy = xy_smoother.update(ids, cx, cy)
            smoothed_centroids[ids] = (int(sx), int(sy))

        # ---------------------------------------------------------
        # B. DEPTH INFERENCE
        # ---------------------------------------------------------
        new_w = 512
        new_h = int(h_orig * (new_w / w_orig))
        small_frame = cv2.resize(frame, (new_w, new_h))
        frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        with torch.no_grad():
            depth_result = depth_estimator(pil_img)
        
        depth_np = np.array(depth_result["depth"]).astype(np.float32)

        # --- THE GRADIENT FIX ---
        # The ground creates a massive Y-gradient. The arm is thin, so the median of each row 
        # perfectly represents the ground's depth at that Y-level.
        # By subtracting it, we isolate the arm's true depth (closeness relative to the ground behind it)
        # completely nullifying the background interference!
        ground_profile = np.median(depth_np, axis=1) # Shape: (new_h,)
        depth_np = depth_np - ground_profile[:, np.newaxis]

        # ---------------------------------------------------------
        # C. EXTRACT & SMOOTH DEPTH
        # ---------------------------------------------------------
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
                
            smoothed_depth = depth_ema.get(ids, 0)
            joint_depths[ids] = smoothed_depth
            
            # Dynamically stretch the color palette specifically to the arm's depth range
            if smoothed_depth < running_depth_min:
                running_depth_min = smoothed_depth
            if smoothed_depth > running_depth_max:
                running_depth_max = smoothed_depth

        # ---------------------------------------------------------
        # D. STORE PRECOMPUTED DATA
        # ---------------------------------------------------------
        # We only store mathematical coordinates, NOT the full image, so memory usage is basically zero!
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
            'gripper_dist': gripper_dist
        })

    print("\n\n--- PASS 2: Playback ---")
    print("Precomputation finished! Starting smooth playback...")
    print("Press 'q' at any time to exit and flush memory.")
    
    cv2.namedWindow("Combined Inference (Lite/Precomputed)", cv2.WINDOW_NORMAL)
    
    # Endless playback loop until 'q' is pressed
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Rewind video
        exit_playback = False
        
        for frame_idx, data in enumerate(precomputed_data):
            ret, frame = cap.read()
            if not ret:
                break
                
            display = frame.copy()
            smoothed_centroids = data['centroids']
            
            # Calculate Colors dynamically using the global joint-only min/max we found in Pass 1
            joint_colors = {}
            for ids, depth_val in data['depths'].items():
                joint_colors[ids] = get_color_from_depth(depth_val, running_depth_min, running_depth_max)

            # Draw Skeleton Lines
            skeleton_links = [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5)]
            for (s, e) in skeleton_links:
                if s in smoothed_centroids and e in smoothed_centroids:
                    cv2.line(display, smoothed_centroids[s], smoothed_centroids[e], (255, 255, 0), 2)

            # Draw Gripper Line and Dynamic Square
            if 4 in smoothed_centroids and 5 in smoothed_centroids:
                x4, y4 = smoothed_centroids[4]
                x5, y5 = smoothed_centroids[5]
                
                # Draw line connecting J4 and J5
                cv2.line(display, (x4, y4), (x5, y5), (255, 255, 255), 2)
                
                mx, my = (x4 + x5) // 2, (y4 + y5) // 2
                g_dist = data['gripper_dist']
                
                # Calculate ratio (0.0 to 1.0)
                if gripper_max_dist > gripper_min_dist:
                    ratio = (g_dist - gripper_min_dist) / (gripper_max_dist - gripper_min_dist)
                else:
                    ratio = 0.5
                
                # Map ratio to BGR color: Wide open (ratio=1) -> Green, Closed (ratio=0) -> Red
                b_val = 0
                g_val = int(255 * ratio)
                r_val = int(255 * (1 - ratio))
                box_color = (b_val, g_val, r_val)
                
                # Draw square (same size as circle radius 7 -> 14x14 square)
                radius = 7
                # Black outline
                cv2.rectangle(display, (mx - radius - 2, my - radius - 2), (mx + radius + 2, my + radius + 2), (0,0,0), -1)
                # Colored fill
                cv2.rectangle(display, (mx - radius, my - radius), (mx + radius, my + radius), box_color, -1)

            # Draw Joints
            for ids, (cx, cy) in smoothed_centroids.items():
                color = joint_colors[ids]
                if ids <= 5:
                    radius = 7
                    cv2.circle(display, (cx, cy), radius + 2, (0, 0, 0), -1)
                    cv2.circle(display, (cx, cy), radius, color, -1)
                    cv2.putText(display, f"J{ids}", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                    cv2.putText(display, f"J{ids}", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                else:
                    radius = 10
                    cv2.circle(display, (cx, cy), radius + 2, (0, 0, 0), -1)
                    cv2.circle(display, (cx, cy), radius, color, -1)
                    cv2.putText(display, "Tgt", (cx + 15, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                    cv2.putText(display, "Tgt", (cx + 15, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.putText(display, "Joint Depth: Bright Yellow = Close | Dark Purple = Far", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Combined Inference (Lite/Precomputed)", display)

            # Playback at ~33fps (30ms wait)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                exit_playback = True
                break
                
        if exit_playback:
            break

    # Clean up and flush memory
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
