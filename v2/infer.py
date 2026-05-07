import argparse
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
import pybullet_data
import torch
import torch.nn as nn


CURRENT_DIR = Path(__file__).resolve().parent
VVA_DIR = CURRENT_DIR.parent
sys.path.append(str(VVA_DIR))

from project_paths import DEMOVIDEO_DIR, ROBOT_URDF, YOLO_DIR, YOLO_WEIGHTS
from ultralytics import YOLO

sys.path.append(str(YOLO_DIR))
from utils.smoothing import CentroidSmoother

sys.path.append(str(CURRENT_DIR))
from extract_points_2d import process_video_2d


INTENT_MODEL_PATH = CURRENT_DIR / "model2_vla_2d_intent.pth"
PHASE_MODEL_PATH = CURRENT_DIR / "model2_vla_2d_phase.pth"
MODEL_PATH = INTENT_MODEL_PATH if INTENT_MODEL_PATH.exists() else PHASE_MODEL_PATH
DATASET_PATH = CURRENT_DIR / "dataset_2d.npy"
H_WINDOW = 10
INTENT_FEATURE_DIM = 24


def points14_to_intent_features(points14, eps=1e-6):
    """Convert absolute YOLO points into robot-relative task geometry."""
    arr = np.asarray(points14, dtype=np.float32)
    if arr.shape[-1] != 14:
        raise ValueError(f"Expected last dimension 14, got {arr.shape[-1]}")

    leading_shape = arr.shape[:-1]
    pts = arr.reshape(-1, 7, 2)

    base = pts[:, 0]
    shoulder = pts[:, 1]
    elbow = pts[:, 2]
    wrist = pts[:, 3]
    grip_l = pts[:, 4]
    grip_r = pts[:, 5]
    target = pts[:, 6]

    ee = 0.5 * (grip_l + grip_r)
    scale = (
        np.linalg.norm(shoulder - base, axis=1)
        + np.linalg.norm(elbow - shoulder, axis=1)
        + np.linalg.norm(wrist - elbow, axis=1)
        + np.linalg.norm(ee - wrist, axis=1)
    )
    scale = np.maximum(scale, eps).reshape(-1, 1)

    centered_points = ((pts - base[:, None, :]) / scale[:, None, :]).reshape(-1, 14)
    ee_rel_base = (ee - base) / scale
    target_rel_ee = (target - ee) / scale
    grip_vec = (grip_r - grip_l) / scale

    grip_width = np.linalg.norm(grip_r - grip_l, axis=1, keepdims=True) / scale
    target_dist = np.linalg.norm(target - ee, axis=1, keepdims=True) / scale
    target_dir = target_rel_ee / np.maximum(target_dist, eps)

    features = np.concatenate(
        [
            centered_points,
            ee_rel_base,
            target_rel_ee,
            grip_vec,
            grip_width,
            target_dist,
            target_dir,
        ],
        axis=1,
    ).astype(np.float32)

    return features.reshape(*leading_shape, INTENT_FEATURE_DIM)


def sequence_to_intent_features(sequence):
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.size == 0:
        return np.empty((0, INTENT_FEATURE_DIM), dtype=np.float32)
    return points14_to_intent_features(sequence)


def transform_for_policy(features, feature_mode):
    if feature_mode == "intent_relative":
        return sequence_to_intent_features(features)
    if feature_mode == "absolute":
        return np.asarray(features, dtype=np.float32)
    raise ValueError(f"Unsupported feature_mode in checkpoint: {feature_mode}")


def phase_features(progress):
    pval = progress.clamp(0.0, 1.0)
    return torch.cat(
        [
            pval,
            torch.sin(math.pi * pval),
            torch.cos(math.pi * pval),
            torch.sin(2.0 * math.pi * pval),
            torch.cos(2.0 * math.pi * pval),
        ],
        dim=1,
    )


class PromptPolicy2D(nn.Module):
    def __init__(
        self,
        input_dim=14,
        h_window=10,
        phase_dim=5,
        embed_dim=160,
        action_dim=6,
        nhead=4,
        num_layers=4,
        dropout=0.08,
        max_tokens=2000,
    ):
        super().__init__()
        self.max_tokens = max_tokens

        self.demo_embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.state_embed = nn.Sequential(
            nn.Linear(input_dim * h_window + phase_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.pos_emb = nn.Parameter(torch.randn(1, max_tokens, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.action_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 160),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(160, 96),
            nn.GELU(),
            nn.Linear(96, action_dim),
        )

    def forward(self, demo_seq, demo_mask, cam_hist, progress):
        batch_size, seq_len, _ = demo_seq.shape
        if seq_len + 1 > self.max_tokens:
            raise ValueError(f"Demo sequence is too long: {seq_len}, max is {self.max_tokens - 1}")

        demo_tokens = self.demo_embed(demo_seq)
        demo_tokens = demo_tokens + self.pos_emb[:, 1:seq_len + 1, :]

        state_in = torch.cat([cam_hist, phase_features(progress)], dim=1)
        state_token = self.state_embed(state_in).unsqueeze(1)
        state_token = state_token + self.pos_emb[:, :1, :]

        full_seq = torch.cat([state_token, demo_tokens], dim=1)
        state_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=demo_mask.device)
        full_mask = torch.cat([state_mask, demo_mask], dim=1)

        out_seq = self.transformer(full_seq, src_key_padding_mask=full_mask)
        return self.action_head(out_seq[:, 0, :])


def parse_demo_number(path):
    match = re.search(r"demovid(\d+)\.mp4$", Path(path).name)
    return int(match.group(1)) if match else None


def numeric_demo_key(path):
    number = parse_demo_number(path)
    return number if number is not None else 10**9


def load_checkpoint(path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise RuntimeError(f"{path} is not a supported v2 checkpoint.")

    config = dict(checkpoint.get("config", {}))
    feature_mode = config.pop("feature_mode", "absolute")
    config.pop("model_class", None)
    config.pop("raw_point_dim", None)
    config.pop("intent_feature_dim", None)

    model = PromptPolicy2D(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    action_mean = checkpoint["action_mean"].to(device).float()
    action_std = checkpoint["action_std"].to(device).float()
    h_window = int(config.get("h_window", H_WINDOW))
    return model, action_mean, action_std, feature_mode, h_window


def load_trial_lengths(dataset_path):
    trial_lengths = {}
    if not dataset_path.exists():
        return trial_lengths

    raw_data = np.load(dataset_path, allow_pickle=True)
    for index, trial in enumerate(raw_data, start=1):
        trial_number = int(trial.get("trial", index))
        cam_len = len(trial.get("cam_X", []))
        action_len = len(trial.get("actions", []))
        if cam_len and action_len:
            trial_lengths[trial_number] = min(cam_len, action_len)
    return trial_lengths


def choose_demo(args, trial_lengths):
    demos = sorted(Path(DEMOVIDEO_DIR).glob("demovid*.mp4"), key=numeric_demo_key)
    if not demos:
        raise FileNotFoundError(f"No demovid*.mp4 files found in {DEMOVIDEO_DIR}")

    if args.demo is not None:
        for demo_path in demos:
            if parse_demo_number(demo_path) == args.demo:
                return demo_path
        raise FileNotFoundError(f"demovid{args.demo}.mp4 was not found in {DEMOVIDEO_DIR}")

    print("\nAvailable demos:")
    for row, demo_path in enumerate(demos, start=1):
        demo_number = parse_demo_number(demo_path)
        if demo_number in trial_lengths:
            note = f"trained trial, rollout {trial_lengths[demo_number]} steps"
        else:
            note = "no paired camview/jointdata in dataset"
        print(f"{row}: {demo_path.name} ({note})")

    choice = int(input("Select demo row number: ").strip()) - 1
    if choice < 0 or choice >= len(demos):
        raise ValueError("Invalid demo choice.")
    return demos[choice]


def estimate_rollout_steps(demo_path, demo_features, trial_lengths):
    demo_number = parse_demo_number(demo_path)
    if demo_number in trial_lengths:
        return trial_lengths[demo_number]
    return max(1, len(demo_features))


def get_robot_view(renderer):
    width, height = 640, 480
    view = p.computeViewMatrixFromYawPitchRoll([0.2, 0, 0.1], 1.2, 45, -40, 0, 2)
    proj = p.computeProjectionMatrixFOV(50, width / height, 0.1, 4.0)
    _, _, rgb, _, _ = p.getCameraImage(
        width,
        height,
        view,
        proj,
        shadow=0,
        renderer=renderer,
    )
    frame = np.reshape(rgb, (height, width, 4))[:, :, :3].astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def create_flat_grey_floor():
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[3.0, 3.0, 0.01])
    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[3.0, 3.0, 0.01],
        rgbaColor=[0.45, 0.45, 0.45, 1.0],
    )
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=[0, 0, -0.01],
    )


def draw_camera_overlay(frame, centroids, progress):
    for idx, pos in centroids.items():
        if pos is None:
            continue
        x, y = int(pos[0]), int(pos[1])
        cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)
        cv2.putText(frame, str(idx), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(
        frame,
        f"phase {progress:.3f}",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )
    return frame


def build_args():
    parser = argparse.ArgumentParser(description="Run v2 VVA inference in PyBullet.")
    parser.add_argument("--demo", type=int, default=None, help="Use demovidN.mp4 directly, for example --demo 5")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to a v2 phase or intent checkpoint")
    parser.add_argument("--phase-speed", type=float, default=1.0, help="Progress multiplier. Lower is slower, e.g. 0.75")
    parser.add_argument("--physics-steps", type=int, default=6, help="PyBullet simulation steps per policy step")
    parser.add_argument("--yolo-conf", type=float, default=0.065, help="YOLO confidence threshold")
    parser.add_argument("--no-camera-window", action="store_true", help="Do not open the OpenCV robot camera window")
    parser.add_argument("--require-cuda", action="store_true", help="Exit immediately if CUDA is not available")
    parser.add_argument(
        "--gpu-sentinel-mb",
        type=int,
        default=0,
        help="Reserve this many MiB on CUDA so nvidia-smi has an obvious live allocation.",
    )
    parser.add_argument("--direct", action="store_true", help="Run PyBullet without opening the GUI")
    parser.add_argument("--max-steps", type=int, default=0, help="Exit after N outer simulation loop steps. 0 runs until quit.")
    parser.add_argument(
        "--renderer",
        choices=("auto", "opengl", "tiny"),
        default="auto",
        help="Camera renderer. auto uses TinyRenderer in --direct mode and OpenGL in GUI mode.",
    )
    return parser.parse_args()


def main():
    args = build_args()
    print(f"Python: {sys.executable}")
    print(f"Torch: {torch.__version__} | CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()} | devices: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")
    elif args.require_cuda:
        print("CUDA was required but is not available. Run .\\setup_windows.ps1 -Cuda from C:\\vvaw.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    gpu_sentinel = None
    if device.type == "cuda" and args.gpu_sentinel_mb > 0:
        sentinel_elements = max(1, args.gpu_sentinel_mb * 1024 * 1024 // 4)
        gpu_sentinel = torch.empty(sentinel_elements, dtype=torch.float32, device=device)
        gpu_sentinel.fill_(1.0)
        torch.cuda.synchronize()
        print(f"Reserved {args.gpu_sentinel_mb} MiB CUDA sentinel for nvidia-smi visibility.")

    trial_lengths = load_trial_lengths(DATASET_PATH)
    try:
        demo_path = choose_demo(args, trial_lengths)
    except Exception as exc:
        print(f"Could not select demo: {exc}")
        return

    if not args.model.exists():
        print(f"Model not found: {args.model}")
        print("Expected model2_vla_2d_intent.pth or model2_vla_2d_phase.pth in this v2 folder.")
        return

    print("Loading YOLO...")
    yolo_model = YOLO(str(YOLO_WEIGHTS)).to(device)

    print(f"Loading VVA policy: {args.model}")
    try:
        model, action_mean, action_std, feature_mode, h_window = load_checkpoint(args.model, device)
    except Exception as exc:
        print(f"Could not load policy: {exc}")
        return
    print(f"Feature mode: {feature_mode}")

    print(f"Processing visual prompt: {demo_path.name}")
    raw_demo_features = process_video_2d(str(demo_path), yolo_model)
    if raw_demo_features is None or len(raw_demo_features) == 0:
        print("Failed to process selected demo.")
        return
    demo_features = transform_for_policy(raw_demo_features, feature_mode)

    rollout_steps = estimate_rollout_steps(demo_path, raw_demo_features, trial_lengths)
    print(f"Using rollout length: {rollout_steps} policy steps")
    print(f"Phase speed: {args.phase_speed}")

    demo_seq_tensor = torch.tensor(demo_features, dtype=torch.float32).unsqueeze(0).to(device)
    demo_mask_tensor = torch.zeros(1, len(demo_features), dtype=torch.bool, device=device)

    print("Starting PyBullet simulation...")
    connect_mode = p.DIRECT if args.direct else p.GUI
    p.connect(connect_mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    if not args.direct:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.1,
            cameraYaw=45,
            cameraPitch=-35,
            cameraTargetPosition=[0.2, 0, 0.1],
        )
    if args.renderer == "tiny" or (args.renderer == "auto" and args.direct):
        renderer = p.ER_TINY_RENDERER
    else:
        renderer = p.ER_BULLET_HARDWARE_OPENGL
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 120.0)

    create_flat_grey_floor()
    robot = p.loadURDF(str(ROBOT_URDF), basePosition=[0, 0, 0], useFixedBase=True)
    cube_id = p.loadURDF("cube.urdf", basePosition=[0.4, 0.0, 0.05], globalScaling=0.05)
    p.changeVisualShape(cube_id, -1, rgbaColor=[1, 0, 0, 1])

    joints = [0, 1, 2, 3, 4, 5]
    for joint in joints:
        p.resetJointState(robot, joint, 0)

    last_known_centroids = {}
    xy_smoother = CentroidSmoother()
    persistent_centroids = {idx: None for idx in range(7)}
    cam_history = []
    initialized = False
    policy_step = 0
    loop_step = 0
    show_camera = not args.no_camera_window and not args.direct

    limits = [
        (-3.14, 3.14),
        (-3.14, 3.14),
        (-3.14, 3.14),
        (-3.14, 3.14),
        (0.0, 0.5),
        (-0.5, 0.0),
    ]

    print("VVA inference active. Press q in the camera window or Ctrl+C to quit.")

    try:
        while True:
            frame = get_robot_view(renderer)
            img_h, img_w = frame.shape[:2]

            yolo_results = yolo_model(frame, verbose=False, conf=args.yolo_conf)
            current_centroids = {}

            if yolo_results[0].boxes:
                for box in yolo_results[0].boxes:
                    idx = int(box.cls[0].item())
                    x, y, _, _ = box.xywh[0].cpu().numpy()
                    cx, cy = int(x), int(y)
                    if idx in last_known_centroids:
                        lx, ly = last_known_centroids[idx]
                        if ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5 > 40:
                            continue
                    current_centroids[idx] = (cx, cy)

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

            for idx in range(7):
                if idx not in current_centroids and idx in last_known_centroids:
                    current_centroids[idx] = last_known_centroids[idx]
            for idx, pos in current_centroids.items():
                last_known_centroids[idx] = pos

            smoothed_centroids = {}
            for idx, (cx, cy) in current_centroids.items():
                sx, sy = xy_smoother.update(idx, cx, cy)
                smoothed_centroids[idx] = (sx, sy)

            for idx in range(7):
                if idx in smoothed_centroids:
                    persistent_centroids[idx] = smoothed_centroids[idx]

            progress_value = min((policy_step * args.phase_speed) / max(1, rollout_steps - 1), 1.0)

            if not initialized:
                if all(pos is not None for pos in persistent_centroids.values()):
                    initialized = True
                    print("All 7 points initialized. Starting policy.")
                else:
                    for _ in range(args.physics_steps):
                        p.stepSimulation()
                    loop_step += 1
                    if show_camera:
                        overlay = draw_camera_overlay(frame.copy(), persistent_centroids, progress_value)
                        cv2.imshow("Robot Camera View", overlay)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    if args.max_steps and loop_step >= args.max_steps:
                        print(f"\nReached --max-steps={args.max_steps} before all points initialized.")
                        break
                    continue

            current_cam_features = np.zeros(14, dtype=np.float32)
            for idx in range(7):
                cx, cy = persistent_centroids[idx]
                current_cam_features[idx * 2] = cx / img_w
                current_cam_features[idx * 2 + 1] = cy / img_h

            policy_cam_features = transform_for_policy(current_cam_features, feature_mode)

            cam_history.append(policy_cam_features.copy())
            if len(cam_history) > h_window:
                cam_history.pop(0)

            if len(cam_history) < h_window:
                hist_array = np.vstack([np.tile(cam_history[0], (h_window - len(cam_history), 1)), cam_history])
            else:
                hist_array = np.asarray(cam_history, dtype=np.float32)

            cam_tensor = torch.tensor(hist_array.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
            progress_tensor = torch.tensor([[progress_value]], dtype=torch.float32, device=device)

            with torch.no_grad():
                pred_normalized = model(demo_seq_tensor, demo_mask_tensor, cam_tensor, progress_tensor)
                pred_actions = pred_normalized * action_std + action_mean
                target_pos = pred_actions[0].detach().cpu().numpy()

            for joint_idx, (lo, hi) in enumerate(limits):
                target_pos[joint_idx] = np.clip(target_pos[joint_idx], lo, hi)

            p.setJointMotorControlArray(
                robot,
                joints,
                p.POSITION_CONTROL,
                targetPositions=target_pos,
                forces=[100] * len(joints),
            )

            for _ in range(args.physics_steps):
                p.stepSimulation()

            if policy_step % 30 == 0:
                print(f"\rstep {policy_step:04d} | phase {progress_value:.3f} | target {np.round(target_pos, 3)}", end="")

            if show_camera:
                overlay = draw_camera_overlay(frame.copy(), persistent_centroids, progress_value)
                cv2.imshow("Robot Camera View", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            policy_step += 1
            loop_step += 1
            if args.max_steps and loop_step >= args.max_steps:
                print(f"\nReached --max-steps={args.max_steps}.")
                break

    except KeyboardInterrupt:
        print("\nStopping inference.")
    finally:
        cv2.destroyAllWindows()
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
