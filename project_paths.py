import os
from pathlib import Path

VVA_ROOT = Path(__file__).resolve().parent
SUBSYSTEMS_DIR = VVA_ROOT / "subsystems"
CACHE_DIR = VVA_ROOT / ".cache"

RUNTIME_ENV_DEFAULTS = {
    "YOLO_CONFIG_DIR": CACHE_DIR / "ultralytics",
    "HF_HOME": CACHE_DIR / "huggingface",
    "TORCH_HOME": CACHE_DIR / "torch",
    "MPLCONFIGDIR": CACHE_DIR / "matplotlib",
    "PIP_CACHE_DIR": CACHE_DIR / "pip",
}

for name, path in RUNTIME_ENV_DEFAULTS.items():
    path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(name, str(path))

DATA_DIR = VVA_ROOT / "data"
CAMVIEW_DIR = DATA_DIR / "camview"
DEMOVIDEO_DIR = DATA_DIR / "demovideos"
RAWDATA_JOINT_DIR = DATA_DIR / "jointdata"

YOLO_DIR = VVA_ROOT / "yolo"
YOLO_WEIGHTS = YOLO_DIR / "models" / "run" / "weights" / "best.pt"

ASSETS_DIR = VVA_ROOT / "assets"
URDF_DIR = ASSETS_DIR / "urdf"
ROBOT_URDF = URDF_DIR / "gripper_arm.urdf"

DATASET_3D = VVA_ROOT / "dataset.npy"


def approach_dir(name: str) -> Path:
    return VVA_ROOT / name
