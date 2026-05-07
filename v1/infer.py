import os
import sys
import cv2
import torch
import numpy as np
import pybullet as p
import pybullet_data
import warnings
from PIL import Image

# --- SUPPRESS WARNINGS ---
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.pipelines")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# --- CONFIG ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VVA_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(VVA_DIR)
from project_paths import DEMOVIDEO_DIR, ROBOT_URDF, YOLO_DIR, YOLO_WEIGHTS
from transformers import pipeline
from ultralytics import YOLO
sys.path.append(str(YOLO_DIR))
from utils.smoothing import CentroidSmoother

MODEL_PATH = os.path.join(CURRENT_DIR, "model1_vla.pth")
DEMO_DIR = str(DEMOVIDEO_DIR)

# Import the model architecture
sys.path.append(CURRENT_DIR)
from train1 import VLAPromptPolicy

def get_robot_view():
    # Standard 640x480 resolution for YOLO consistency
    w, h = 640, 480
    view = p.computeViewMatrixFromYawPitchRoll([0.2, 0, 0.1], 1.2, 45, -40, 0, 2)
    proj = p.computeProjectionMatrixFOV(50, w/h, 0.1, 4.0)
    _, _, rgb, _, _ = p.getCameraImage(w, h, view, proj, shadow=0, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    frame = np.reshape(rgb, (h, w, 4))[:, :, :3].astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def process_video_for_demo(vid_path, yolo_model, depth_estimator):
    """ Runs the exact extraction logic to build the Visual Prompt E_demo """
    print(f"Precomputing Demovideo {vid_path}...")
    cap = cv2.VideoCapture(vid_path)
    
    last_known_centroids = {}
    xy_smoother = CentroidSmoother()
    depth_ema = {}
    EMA_ALPHA = 0.6  
    
    gripper_min_dist = float('inf')
    gripper_max_dist = float('-inf')
    depth_min = float('inf')
    depth_max = float('-inf')

    precomputed_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        h_orig, w_orig = frame.shape[:2]
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
                dx_coord = max(0, min(new_w - 1, int(cx * (new_w / w_orig))))
                dy_coord = max(0, min(new_h - 1, int(cy * (new_h / h_orig))))
                raw_depth = depth_np[dy_coord, dx_coord]
                
                if ids not in depth_ema: depth_ema[ids] = raw_depth
                else: depth_ema[ids] = EMA_ALPHA * raw_depth + (1 - EMA_ALPHA) * depth_ema[ids]
                
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
            'w_orig': w_orig,
            'h_orig': h_orig
        })
    cap.release()

    # Backfill and Build Matrix
    demo_X = []
    if not precomputed_data: return np.array([])
        
    current_features = np.zeros(22)
    found_initial = [False] * 7
    found_initial_gripper = False
    
    for data in precomputed_data:
        centroids = data['centroids']
        for ids in range(7):
            if ids in centroids and not found_initial[ids]:
                nz = 0.0
                if depth_max > depth_min:
                    nz = (data['depths'].get(ids, 0.0) - depth_min) / (depth_max - depth_min)
                current_features[ids*3] = centroids[ids][0] / data['w_orig']
                current_features[ids*3 + 1] = centroids[ids][1] / data['h_orig']
                current_features[ids*3 + 2] = nz
                found_initial[ids] = True
        if 4 in centroids and 5 in centroids and not found_initial_gripper:
            intensity = 0.5
            if gripper_max_dist > gripper_min_dist:
                intensity = (data['gripper_dist'] - gripper_min_dist) / (gripper_max_dist - gripper_min_dist)
            current_features[21] = intensity
            found_initial_gripper = True
        if all(found_initial) and found_initial_gripper: break

    for data in precomputed_data:
        centroids = data['centroids']
        for ids in range(7):
            if ids in centroids:
                nz = 0.0
                if depth_max > depth_min:
                    nz = (data['depths'].get(ids, 0.0) - depth_min) / (depth_max - depth_min)
                current_features[ids*3] = centroids[ids][0] / data['w_orig']
                current_features[ids*3 + 1] = centroids[ids][1] / data['h_orig']
                current_features[ids*3 + 2] = nz
        if 4 in centroids and 5 in centroids:
            intensity = 0.5
            if gripper_max_dist > gripper_min_dist:
                intensity = (data['gripper_dist'] - gripper_min_dist) / (gripper_max_dist - gripper_min_dist)
            current_features[21] = intensity
        demo_X.append(current_features.copy())
            
    return np.array(demo_X)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Ask for Demo Video
    try:
        demos = [f for f in os.listdir(DEMO_DIR) if f.startswith("demovid") and f.endswith(".mp4")]
        demos.sort()
        print("\nAvailable Demos:")
        for i, d in enumerate(demos): print(f"{i+1}: {d}")
        choice = int(input("Select demo number: ")) - 1
        demo_file = os.path.join(DEMO_DIR, demos[choice])
    except:
        print("Invalid choice.")
        return

    # 2. Load Models
    print("Loading YOLO & Depth Anything...")
    yolo_model = YOLO(str(YOLO_WEIGHTS)).to(device)
    depth_estimator = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf", device=0 if torch.cuda.is_available() else -1)
    
    print("Loading Trained VLA Policy...")
    model = VLAPromptPolicy().to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Trained model not found at {MODEL_PATH}. Train it first!")
        return
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 3. Precompute Demo Sequence (Visual Prompt)
    demo_X_np = process_video_for_demo(demo_file, yolo_model, depth_estimator)
    if len(demo_X_np) == 0:
        print("Failed to process demo.")
        return
    
    # Convert to Tensor [1, SeqLen, 22]
    # We do NOT need padding masks because batch size is 1! Attention handles variable sequences dynamically.
    demo_seq_tensor = torch.tensor(demo_X_np, dtype=torch.float32).unsqueeze(0).to(device)
    demo_mask_tensor = torch.zeros(1, demo_X_np.shape[0], dtype=torch.bool).to(device)

    # 4. Setup PyBullet Simulator
    print("Starting Simulation...")
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    robot = p.loadURDF(str(ROBOT_URDF), basePosition=[0, 0, 0], useFixedBase=True)
    cube_id = p.loadURDF("cube.urdf", basePosition=[0.4, 0.0, 0.05], globalScaling=0.05)
    p.changeVisualShape(cube_id, -1, rgbaColor=[1, 0, 0, 1])

    joints = [0, 1, 2, 3, 4, 5]
    for i in joints: p.resetJointState(robot, i, 0)
    
    # 5. Live Inference Loop State
    last_known_centroids = {}
    xy_smoother = CentroidSmoother()
    depth_ema = {}
    EMA_ALPHA = 0.6
    
    # We must track min/max dynamically as they appear, like the old infer script!
    running_depth_min = float('inf')
    running_depth_max = float('-inf')
    running_gripper_min = float('inf')
    running_gripper_max = float('-inf')
    current_cam_features = np.zeros(22)
    
    print("VLA Inference Active. Press Ctrl+C in the terminal to quit.")

    while True:
        # A. Grab Simulator Frame
        frame = get_robot_view()
        h_orig, w_orig = frame.shape[:2]
        
        # B. YOLO Detection
        yolo_results = yolo_model(frame, verbose=False, conf=0.065)
        current_centroids = {}
        
        if yolo_results[0].boxes:
            for box in yolo_results[0].boxes:
                ids = int(box.cls[0].item())
                x, y, w, h = box.xywh[0].cpu().numpy()
                cx, cy = int(x), int(y)
                if ids in last_known_centroids:
                    lx, ly = last_known_centroids[ids]
                    if ((cx - lx)**2 + (cy - ly)**2)**0.5 > 40: continue
                current_centroids[ids] = (cx, cy)

        # Missing Joint Logic
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

        real_detections = set(current_centroids.keys())

        # Persistence
        for i in range(7):
            if i not in current_centroids and i in last_known_centroids:
                current_centroids[i] = last_known_centroids[i]
        for ids, pos in current_centroids.items():
            last_known_centroids[ids] = pos

        # Smoothing
        smoothed_centroids = {}
        for ids, (cx, cy) in current_centroids.items():
            sx, sy = xy_smoother.update(ids, cx, cy)
            smoothed_centroids[ids] = (int(sx), int(sy))

        # C. Depth Anything
        new_w = 512
        new_h = int(h_orig * (new_w / w_orig))
        small_frame = cv2.resize(frame, (new_w, new_h))
        frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        with torch.no_grad():
            depth_result = depth_estimator(Image.fromarray(frame_rgb))
        depth_np = np.array(depth_result["depth"]).astype(np.float32)
        depth_np = depth_np - np.median(depth_np, axis=1)[:, np.newaxis]

        # D. Construct Live 22-Feature Vector
        for ids in range(7):
            if ids in smoothed_centroids:
                cx, cy = smoothed_centroids[ids]
                nx = cx / w_orig
                ny = cy / h_orig
                
                # Update running depth
                if ids in real_detections:
                    dx_coord = max(0, min(new_w - 1, int(cx * (new_w / w_orig))))
                    dy_coord = max(0, min(new_h - 1, int(cy * (new_h / h_orig))))
                    raw_depth = depth_np[dy_coord, dx_coord]
                    if ids not in depth_ema: depth_ema[ids] = raw_depth
                    else: depth_ema[ids] = EMA_ALPHA * raw_depth + (1 - EMA_ALPHA) * depth_ema[ids]
                    
                smoothed_depth = depth_ema.get(ids, 0.0)
                if smoothed_depth < running_depth_min: running_depth_min = smoothed_depth
                if smoothed_depth > running_depth_max: running_depth_max = smoothed_depth
                
                nz = 0.0
                if running_depth_max > running_depth_min:
                    nz = (smoothed_depth - running_depth_min) / (running_depth_max - running_depth_min)
                
                current_cam_features[ids*3] = nx
                current_cam_features[ids*3 + 1] = ny
                current_cam_features[ids*3 + 2] = nz

        if 4 in smoothed_centroids and 5 in smoothed_centroids:
            x4, y4 = smoothed_centroids[4]
            x5, y5 = smoothed_centroids[5]
            dist = ((x4 - x5)**2 + (y4 - y5)**2)**0.5
            if dist < running_gripper_min: running_gripper_min = dist
            if dist > running_gripper_max: running_gripper_max = dist
            
            intensity = 0.5
            if running_gripper_max > running_gripper_min:
                intensity = (dist - running_gripper_min) / (running_gripper_max - running_gripper_min)
            current_cam_features[21] = intensity

        # E. VLA Policy Inference
        cam_tensor = torch.tensor(current_cam_features, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Pass (DemoPrompt, Mask, CurrentState) -> Output Actions
            pred_actions = model(demo_seq_tensor, demo_mask_tensor, cam_tensor)
            target_pos = pred_actions[0].cpu().numpy() # [6]

        # Safety Limits
        LIMITS = [(-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14), (0, 0.5), (-0.5, 0)]
        for j in range(6):
            target_pos[j] = np.clip(target_pos[j], LIMITS[j][0], LIMITS[j][1])

        # Execute on Robot
        p.setJointMotorControlArray(robot, joints, p.POSITION_CONTROL, targetPositions=target_pos, forces=[100]*6)
        p.stepSimulation()



    p.disconnect()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
