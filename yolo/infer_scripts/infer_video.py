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

# Model Path 
MODEL_PATH = os.path.join(YOLO_ROOT, "models", "run", "weights", "best.pt")

VIDEO_DIR = os.path.join(REALVVA_ROOT, "data", "demovideos")

def select_video():
    if len(sys.argv) > 1:
        return sys.argv[1]
    import re
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
    
    videos = sorted([f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")], key=natural_sort_key)
    if not videos:
        print("No videos found in data/demovideos!")
        sys.exit(1)
        
    print("\nAvailable Demo Videos:")
    for i, vid in enumerate(videos):
        print(f"[{i+1}] {vid}")
    
    try:
        choice = input(f"\nSelect video [1-{len(videos)}] (default 1): ").strip()
        if not choice:
            return videos[0]
        idx = int(choice) - 1 # Convert 1-based to 0-based
        if 0 <= idx < len(videos):
            return videos[idx]
        else:
            print("Invalid index, using default.")
            return videos[0]
    except ValueError:
        print("Invalid input, using default.")
        return videos[0]

VIDEO_PATH = os.path.join(VIDEO_DIR, select_video())

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video not found at {VIDEO_PATH}")
        return

    print(f"Loading Model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print(f"Processing Video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)

    cv2.namedWindow("YOLO Detect Inference", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Inference with lower threshold (default is 0.25)
        # This helps detect harder-to-see joints like J4
        results = model(frame, verbose=False, conf=0.05)

        # Custom Visualization: Centroids Only
        display = frame.copy()
        
        # Store centroids for skeleton drawing: {class_id: (x, y)}
        centroids = {}
        
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
        
        # Draw Skeleton Lines (J0 -> J1 -> ... -> J5)
        # We only draw if we have both points
        skeleton_links = [(0,1), (1,2), (2,3), (3,4), (3,5)]
        for (s, e) in skeleton_links:
            if s in centroids and e in centroids:
                cv2.line(display, centroids[s], centroids[e], (255, 255, 0), 2)
                
        cv2.imshow("YOLO Detect Inference", display)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
