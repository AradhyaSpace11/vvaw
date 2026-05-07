import cv2
import os
import numpy as np
import sys

# --- CONFIG ---
# --- CONFIG ---
# --- CONFIG ---
CLASSES = [
    "J0_Base", "J1_Shoulder", "J2_Elbow", "J3_Wrist", "J4_GripL", "J5_GripR", "Target"
]
JOINT_OPTS = [
    "J0: Base (Yellow)",
    "J1: Shoulder (Yellow-Green)",
    "J2: Elbow (Blue-Yellow)",
    "J3: Wrist (Red-Blue)",
    "J4: Grip Left (Purple)",
    "J5: Grip Right (Orange)"
]
SKIP_FRAMES = 10 

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) 
YOLO_ROOT = os.path.dirname(CURRENT_DIR) 
REALVVA_ROOT = os.path.dirname(YOLO_ROOT)
DATA_ROOT = os.path.join(REALVVA_ROOT, "data") 
OUTPUT_ROOT = os.path.join(YOLO_ROOT, "data", "dataset")

# Default Video
video_name = "demovid1.mp4"
if len(sys.argv) > 1:
    video_name = sys.argv[1]

VIDEO_PATH = os.path.join(DATA_ROOT, "demovideos", video_name)
OUTPUT_IMG_DIR = os.path.join(OUTPUT_ROOT, "images", "train")
OUTPUT_LBL_DIR = os.path.join(OUTPUT_ROOT, "labels", "train")

# --- UTILS ---
def create_dirs():
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LBL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_IMG_DIR.replace("train", "val"), exist_ok=True)
    os.makedirs(OUTPUT_LBL_DIR.replace("train", "val"), exist_ok=True)

mouse_pos = (0, 0)
def mouse_callback(event, x, y, flags, param):
    global mouse_pos
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_pos = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        param['clicks'].append((x, y))

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video not found at {VIDEO_PATH}")
        return

    create_dirs()
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    print(f"="*60)
    print(f"YOLO FULL BOX LABELING")
    print(f"Video: {video_name}")
    print(f"-"*60)
    print("INSTRUCTIONS:")
    print("1. DRAW BOX for EACH item (J0 -> J5 -> Target).")
    print("   (Click Top-Left, then Click Bottom-Right)")
    print("   (Space=Skip Item, U=Undo, Q=Quit)")
    print(f"="*60)

    frame_idx = 0
    saved_count = 0
    
    cv2.namedWindow("Full Box Labeling")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if frame_idx % SKIP_FRAMES != 0:
            frame_idx += 1
            continue

        clicks = []
        cv2.setMouseCallback("Full Box Labeling", mouse_callback, {'clicks': clicks})
        
        # State:
        # 0=J0_P1, 1=J0_P2
        # 2=J1_P1, 3=J1_P2
        # ...
        # 12=Target_P1, 13=Target_P2
        
        state = 0   
        current_boxes = [] # List of (class_id, x1, y1, x2, y2)
        
        while True:
            display = frame.copy()
            H, W = frame.shape[:2]
            
            # Determine Current Class
            # State 0,1 -> Class 0 (J0)
            # State 12,13 -> Class 6 (Target)
            current_class = state // 2
            is_second_click = (state % 2 == 1)
            
            if current_class < 6:
                label_name = JOINT_OPTS[current_class]
            else:
                label_name = "TARGET"
                
            action = "Click Bottom-Right" if is_second_click else "Click Top-Left"
            msg = f"{label_name}: {action}"
            
            # Text
            cv2.putText(display, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Draw Completed Boxes
            for (cls, x1, y1, x2, y2) in current_boxes:
                c = (0, 255, 0) if cls < 6 else (0, 0, 255)
                cv2.rectangle(display, (x1, y1), (x2, y2), c, 2)
                lbl = f"J{cls}" if cls < 6 else "Tgt"
                cv2.putText(display, lbl, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

            # Draw Current Box (Ghost)
            if is_second_click and len(clicks) == 0:
                # We have the first point (stored in active_p1, simpler to check clicks or store it)
                # Actually, `clicks` is emptied. We need to store P1 somewhere.
                # Let's change architecture slightly: current_boxes stores completed. 
                # We need a `active_p1` variable.
                pass
            
            # Crosshair
            mx, my = mouse_pos
            cv2.line(display, (0, my), (W, my), (200, 200, 200), 1)
            cv2.line(display, (mx, 0), (mx, H), (200, 200, 200), 1)
            
            # Draw active rect if we have P1
            if is_second_click and 'active_p1' in locals():
                cv2.rectangle(display, active_p1, (mx, my), (255, 255, 0), 1)

            cv2.imshow("Full Box Labeling", display)
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('q'): return
            
            # Skip current class
            if key == ord(' '):
                # If we are at state 0 (J0 P1) -> Jump to state 2 (J1 P1)
                # If we are at state 1 (J0 P2) -> Jump to state 2 (J1 P1) (Abort current box)
                next_state = (current_class + 1) * 2
                state = next_state
                if state > 13: # Done
                     save_and_exit = True # Logic below
                     break 
                continue

            # Undo
            if key == ord('u'):
                if state > 0:
                    if is_second_click:
                        state -= 1 # Go back to P1
                    else:
                        state -= 2 # Go back to prev class P1
                        if current_boxes: current_boxes.pop()
                    if state < 0: state = 0
                continue
            
            # SAVE if done
            if state > 13:
                 # Logic handled inside click loop usually, but here checking state
                 pass

            # Click Handling
            if len(clicks) > 0:
                cx, cy = clicks.pop(0)
                
                if not is_second_click:
                    # First Click
                    active_p1 = (cx, cy)
                    state += 1
                else:
                    # Second Click
                    x1, y1 = active_p1
                    x2, y2 = cx, cy
                    
                    # Normalize coords
                    if x1 > x2: x1, x2 = x2, x1
                    if y1 > y2: y1, y2 = y2, y1
                    
                    current_boxes.append((current_class, x1, y1, x2, y2))
                    state += 1
                    
                    if state > 13:
                         # SAVE
                        img_filename = f"{video_name.split('.')[0]}_frame_{frame_idx:06d}.jpg"
                        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, img_filename), frame)
                        
                        with open(os.path.join(OUTPUT_LBL_DIR, img_filename.replace(".jpg", ".txt")), "w") as f:
                            for (cls, bx1, by1, bx2, by2) in current_boxes:
                                nx = ((bx1+bx2)/2)/W
                                ny = ((by1+by2)/2)/H
                                nw = (bx2-bx1)/W
                                nh = (by2-by1)/H
                                f.write(f"{cls} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")
                                
                        print(f"Saved {img_filename}")
                        saved_count += 1
                        break
        
        frame_idx += 1

    print(f"Done. Saved {saved_count} frames.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
