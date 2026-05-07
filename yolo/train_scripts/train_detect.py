import os
import sys
import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_ROOT = os.path.dirname(CURRENT_DIR)
REALVVA_ROOT = os.path.dirname(YOLO_ROOT)
sys.path.append(REALVVA_ROOT)
import project_paths  # noqa: F401

from ultralytics import YOLO

# CONFIG
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_ROOT = os.path.dirname(CURRENT_DIR)
DATASET_DIR = os.path.join(YOLO_ROOT, "data", "dataset")
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")
MODEL_OUT_DIR = os.path.join(YOLO_ROOT, "models")
BASE_MODEL = os.path.join(YOLO_ROOT, "models", "base", "yolov8n.pt")

def train():
    # 1. Update data.yaml with ABSOLUTE path
    with open(DATA_YAML, 'r') as f:
        config = yaml.safe_load(f)
    
    config['path'] = DATASET_DIR
    
    with open(DATA_YAML, 'w') as f:
        yaml.dump(config, f)
        
    print(f"Updated config: {DATASET_DIR}")

    # 2. Train using yolov8n.pt (Detection model, NOT Pose)
    model = YOLO(BASE_MODEL)
    
    print(f"Starting Training (Detection)...")
    results = model.train(data=DATA_YAML, epochs=100, imgsz=640, project=MODEL_OUT_DIR, name="run")

if __name__ == "__main__":
    train()
