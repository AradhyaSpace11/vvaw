import pybullet as p
import pybullet_data
import numpy as np
import cv2
import torch
import torch.nn as nn
import os
import sys
import math
import time

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
ROBOT_URDF = os.path.join(REALVVA_ROOT, "assets", "urdf", "gripper_arm.urdf")

# Paths
YOLO_MODEL_PATH = os.path.join(YOLO_ROOT, "models", "run", "weights", "best.pt")
POLICY_MODEL_PATH = os.path.join(YOLO_ROOT, "models", "policy_mlp_detect.pth")

# Default Video
video_name = "demovid1.mp4"
if len(sys.argv) > 1:
    video_name = sys.argv[1]
    
VIDEO_PATH = os.path.join(REALVVA_ROOT, "data", "demovideos", video_name)

# --- MODELS ---
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

# --- SMOOTHING ---
class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = x0
        self.dx_prev = np.zeros_like(x0)
        self.t_prev = t0

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0: return self.x_prev
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

class CentroidSmoother:
    def __init__(self):
        self.filters = {} 
        self.cfg_min_cutoff = 0.5
        self.cfg_beta = 0.05
        self.start_time = time.time()
        
    def update(self, cls_id, cx, cy):
        t = time.time() - self.start_time
        curr = np.array([cx, cy], dtype=np.float32)
        if cls_id not in self.filters:
            self.filters[cls_id] = OneEuroFilter(t, curr, self.cfg_min_cutoff, self.cfg_beta)
            return cx, cy
        res = self.filters[cls_id](t, curr)
        return res[0], res[1]

def main():
    if not os.path.exists(YOLO_MODEL_PATH):
        print("YOLO Model not found.")
        return
    if not os.path.exists(VIDEO_PATH):
        print(f"Video not found: {VIDEO_PATH}")
        return

    print(f"Running Open-Loop Replay on: {video_name}")
    
    yolo_model = YOLO(YOLO_MODEL_PATH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_model = RobotPolicy().to(device)
    policy_model.load_state_dict(torch.load(POLICY_MODEL_PATH, map_location=device))
    policy_model.eval()

    # PyBullet Setup
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    plane_id = p.loadURDF("plane.urdf")
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.3, 0.3, 0.3, 1])
    robot = p.loadURDF(ROBOT_URDF, basePosition=[0, 0, 0], useFixedBase=True)
    
    # We can hide the cube since the robot is reacting to the VIDEO, not the sim layout
    # But let's put one there for reference
    cube_id = p.loadURDF("cube.urdf", basePosition=[0.4, 0, 0.05], globalScaling=0.05)
    p.changeVisualShape(cube_id, -1, rgbaColor=[1, 0, 0, 0.5]) # Semi-transparent
    
    joints = [0, 1, 2, 3, 4, 5]
    smoother = CentroidSmoother()
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # Adjust layout
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0,0,0])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video Finished.")
            break
            
        # 1. Vision (From Video File)
        results = yolo_model(frame, verbose=False, conf=0.05)
        
        state_vec = np.zeros(14)
        found_mask = [False] * 7
        
        display = frame.copy()
        
        if results[0].boxes:
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                if cls_id < 7 and not found_mask[cls_id]:
                    x, y, w, h = box.xywh[0].cpu().numpy()
                    
                    # Smooth
                    sx, sy = smoother.update(cls_id, x, y)
                    
                    # Norm
                    H, W = frame.shape[:2]
                    state_vec[cls_id*2] = sx / W
                    state_vec[cls_id*2+1] = sy / H
                    found_mask[cls_id] = True
                    
                    # Vis
                    col = (0, 255, 0) if cls_id < 6 else (0, 0, 255)
                    cv2.circle(display, (int(sx), int(sy)), 5, col, -1)
        
        # 2. Policy
        if all(found_mask[0:4]):
            X_tensor = torch.tensor(state_vec, dtype=torch.float32).to(device).unsqueeze(0)
            with torch.no_grad():
                action = policy_model(X_tensor).cpu().numpy()[0]
            
            # 3. Actions
            p.setJointMotorControlArray(robot, joints, p.POSITION_CONTROL, targetPositions=action, forces=[150]*4 + [60]*2)
        
        p.stepSimulation()
        
        # Resize display for easier viewing side-by-side
        display_half = cv2.resize(display, (0,0), fx=0.6, fy=0.6)
        cv2.imshow("Video Input", display_half)
        
        if cv2.waitKey(30) & 0xFF == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()
    p.disconnect()

if __name__ == "__main__":
    main()
