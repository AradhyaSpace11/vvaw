import cv2
import os
import sys
import numpy as np
import time
import math

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
MODEL_PATH = os.path.join(YOLO_ROOT, "models", "run", "weights", "best.pt")

# --- ONE EURO FILTER ---
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
    def __init__(self, num_classes=7):
        self.filters = {} # Map ID -> Filter
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

# --- MAIN ---
video_name = "demovid1.mp4"
if len(sys.argv) > 1: video_name = sys.argv[1]
VIDEO_PATH = os.path.join(REALVVA_ROOT, "data", "demovideos", video_name)

def main():
    if not os.path.exists(MODEL_PATH):
        print("Model not found")
        return
        
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    smoother = CentroidSmoother()
    
    cv2.namedWindow("Smoothed Detect", cv2.WINDOW_NORMAL)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        results = model(frame, verbose=False, conf=0.05)
        display = frame.copy()
        
        centroids = {}
        
        if results[0].boxes:
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                x, y, w, h = box.xywh[0].cpu().numpy()
                
                # Smooth
                sx, sy = smoother.update(cls_id, x, y)
                centroids[cls_id] = (int(sx), int(sy))
                
                # Draw
                color = (0, 255, 0) if cls_id < 6 else (0, 0, 255)
                cv2.circle(display, (int(sx), int(sy)), 5, color, -1)
                
        # Links
        links = [(0,1), (1,2), (2,3), (3,4), (3,5)]
        for (s,e) in links:
            if s in centroids and e in centroids:
                cv2.line(display, centroids[s], centroids[e], (255,255,0), 2)
                
        cv2.imshow("Smoothed Detect", display)
        if cv2.waitKey(30) & 0xFF == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
