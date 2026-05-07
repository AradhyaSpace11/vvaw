import cv2
import os
import sys

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

# Model Path (The one doing the object detection)
MODEL_PATH = os.path.join(YOLO_ROOT, "models", "run", "weights", "best.pt")

# Video Source
video_name = "demovid1.mp4" # Default
if len(sys.argv) > 1:
    video_name = sys.argv[1]

VIDEO_PATH = os.path.join(REALVVA_ROOT, "data", "demovideos", video_name)

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video not found at {VIDEO_PATH}")
        return

    print(f"Loading Detection Model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print(f"Processing Video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)

    cv2.namedWindow("YOLO Detect Check", cv2.WINDOW_NORMAL)
    
    # Store centroids for skeleton drawing: {class_id: (x, y)}
    # Persist across frames to reduce flicker (Zero-Order Hold)
    centroids = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Inference
        # conf=0.05 as requested for best results
        results = model(frame, verbose=False, conf=0.05)
        
        display = frame.copy()
        
        # Centroids persist from previous frames
        
        if results[0].boxes:
            for box in results[0].boxes:
                # Get Box Info
                ids = int(box.cls[0].item())
                x, y, w, h = box.xywh[0].cpu().numpy() # Center coords
                
                # Draw Centroid
                # Classes 0-5 are joints, 6 is Target
                if ids <= 5:
                    color = (0, 255, 0) # Green for Joints
                    radius = 5
                    cv2.putText(display, f"J{ids}", (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                else: 
                    color = (0, 0, 255) # Red for Target
                    radius = 8
                    cv2.putText(display, "Tgt", (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                cv2.circle(display, (int(x), int(y)), radius, color, -1)
                centroids[ids] = (int(x), int(y))
        
        # Draw Skeleton Links (J0 -> J1 -> ... -> J5)
        skeleton_links = [(0,1), (1,2), (2,3), (3,4), (3,5)]
        for (s, e) in skeleton_links:
            if s in centroids and e in centroids:
                cv2.line(display, centroids[s], centroids[e], (255, 255, 0), 2)

        cv2.imshow("YOLO Detect Check", display)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
