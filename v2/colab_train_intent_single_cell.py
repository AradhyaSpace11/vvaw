# Paste this entire file into one Google Colab cell.
# Upload dataset_2d.npy when Colab asks for it.
#
# Output:
#   model2_vla_2d_intent.pth
#
# This version is closer to the VVA research goal: it converts absolute
# YOLO points into relative task geometry before training. That makes the
# prompt less dependent on exact pixel position, camera framing, and arm size.

import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

try:
    from google.colab import files
    IN_COLAB = True
except Exception:
    files = None
    IN_COLAB = False


SEED = 42
DATA_PATH = Path("dataset_2d.npy")
MODEL_OUT = Path("model2_vla_2d_intent.pth")

RAW_POINT_DIM = 14
INTENT_FEATURE_DIM = 24
INPUT_DIM = INTENT_FEATURE_DIM
ACTION_DIM = 6
H_WINDOW = 10
PHASE_DIM = 5

EMBED_DIM = 160
NHEAD = 4
NUM_LAYERS = 4
DROPOUT = 0.08

BATCH_SIZE = 64
EPOCHS = 700
LR = 3e-4
WEIGHT_DECAY = 1e-4
CAM_NOISE_STD = 0.012
DEMO_NOISE_STD = 0.018

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


if not DATA_PATH.exists():
    if not IN_COLAB:
        raise FileNotFoundError("dataset_2d.npy was not found in the current folder.")
    print("Upload dataset_2d.npy now.")
    uploaded = files.upload()
    if not uploaded:
        raise RuntimeError("No file was uploaded.")
    DATA_PATH = Path(next(iter(uploaded.keys())))

raw_data = np.load(DATA_PATH, allow_pickle=True)
print(f"Loaded {len(raw_data)} trials from {DATA_PATH}")


def as_float_array(x):
    return np.asarray(x, dtype=np.float32)


def points14_to_intent_features(points14, eps=1e-6):
    arr = np.asarray(points14, dtype=np.float32)
    if arr.shape[-1] != RAW_POINT_DIM:
        raise ValueError(f"Expected last dimension {RAW_POINT_DIM}, got {arr.shape[-1]}")

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


class IntentVLADataset2D(Dataset):
    def __init__(self, trials, h_window=10):
        self.h_window = h_window
        self.samples = []

        all_actions = []
        converted_trials = []

        for trial in trials:
            demo = sequence_to_intent_features(as_float_array(trial["demo_X"]))
            cam = sequence_to_intent_features(as_float_array(trial["cam_X"]))
            actions = as_float_array(trial["actions"])
            converted_trials.append((trial, demo, cam, actions))
            all_actions.append(actions)

        self.max_demo_len = max(len(demo) for _, demo, _, _ in converted_trials)

        all_actions = np.concatenate(all_actions, axis=0)
        self.action_mean = all_actions.mean(axis=0).astype(np.float32)
        self.action_std = np.maximum(all_actions.std(axis=0), 1e-4).astype(np.float32)

        for trial_id, (trial, demo, cam, actions) in enumerate(converted_trials):
            n = min(len(cam), len(actions))
            if len(demo) == 0 or n == 0:
                continue

            pad_len = self.max_demo_len - len(demo)
            demo_padded = np.pad(demo, ((0, pad_len), (0, 0)), mode="constant")
            demo_mask = np.zeros(self.max_demo_len, dtype=bool)
            demo_mask[len(demo):] = True

            for t in range(n):
                if t < h_window:
                    prefix = np.tile(cam[0], (h_window - t - 1, 1))
                    hist = np.vstack([prefix, cam[:t + 1]])
                else:
                    hist = cam[t - h_window + 1:t + 1]

                progress = 0.0 if n <= 1 else t / float(n - 1)
                target = (actions[t] - self.action_mean) / self.action_std

                self.samples.append({
                    "demo_seq": torch.tensor(demo_padded, dtype=torch.float32),
                    "demo_mask": torch.tensor(demo_mask, dtype=torch.bool),
                    "cam_hist": torch.tensor(hist.flatten(), dtype=torch.float32),
                    "progress": torch.tensor([progress], dtype=torch.float32),
                    "action": torch.tensor(target, dtype=torch.float32),
                    "trial_id": trial_id,
                    "trial_number": int(trial.get("trial", trial_id + 1)),
                })

        print("Feature mode: intent_relative")
        print("Feature dim:", INTENT_FEATURE_DIM)
        print("Max demo length:", self.max_demo_len)
        print("Samples:", len(self.samples))
        print("Action mean:", self.action_mean)
        print("Action std:", self.action_std)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def phase_features(progress):
    p = progress.clamp(0.0, 1.0)
    return torch.cat([
        p,
        torch.sin(math.pi * p),
        torch.cos(math.pi * p),
        torch.sin(2.0 * math.pi * p),
        torch.cos(2.0 * math.pi * p),
    ], dim=1)


class IntentPromptPolicy2D(nn.Module):
    def __init__(
        self,
        input_dim=24,
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

        demo_tokens = self.demo_embed(demo_seq) + self.pos_emb[:, 1:seq_len + 1, :]
        state_in = torch.cat([cam_hist, phase_features(progress)], dim=1)
        state_token = self.state_embed(state_in).unsqueeze(1) + self.pos_emb[:, :1, :]

        full_seq = torch.cat([state_token, demo_tokens], dim=1)
        state_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=demo_mask.device)
        full_mask = torch.cat([state_mask, demo_mask], dim=1)

        out = self.transformer(full_seq, src_key_padding_mask=full_mask)
        return self.action_head(out[:, 0, :])


dataset = IntentVLADataset2D(raw_data, h_window=H_WINDOW)

val_len = max(1, int(0.10 * len(dataset)))
train_len = len(dataset) - val_len
train_ds, val_ds = random_split(
    dataset,
    [train_len, val_len],
    generator=torch.Generator().manual_seed(SEED),
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = IntentPromptPolicy2D(
    input_dim=INPUT_DIM,
    h_window=H_WINDOW,
    phase_dim=PHASE_DIM,
    embed_dim=EMBED_DIM,
    action_dim=ACTION_DIM,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    max_tokens=max(2000, dataset.max_demo_len + 1),
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
loss_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.2, 2.2], device=device)


def batch_loss(batch, train_mode):
    demo_seq = batch["demo_seq"].to(device)
    demo_mask = batch["demo_mask"].to(device)
    cam_hist = batch["cam_hist"].to(device)
    progress = batch["progress"].to(device)
    target = batch["action"].to(device)

    if train_mode:
        if CAM_NOISE_STD > 0:
            cam_hist = cam_hist + torch.randn_like(cam_hist) * CAM_NOISE_STD
        if DEMO_NOISE_STD > 0:
            valid = (~demo_mask).unsqueeze(-1).float()
            demo_seq = demo_seq + torch.randn_like(demo_seq) * DEMO_NOISE_STD * valid
        progress = (progress + torch.randn_like(progress) * 0.006).clamp(0.0, 1.0)

    pred = model(demo_seq, demo_mask, cam_hist, progress)
    return ((pred - target) ** 2 * loss_weights).mean()


best_val = float("inf")
best_state = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad(set_to_none=True)
        loss = batch_loss(batch, train_mode=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()
    train_loss /= max(1, len(train_loader))

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            val_loss += batch_loss(batch, train_mode=False).item()
    val_loss /= max(1, len(val_loader))

    if val_loss < best_val:
        best_val = val_loss
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if epoch == 1 or epoch % 25 == 0:
        print(
            f"Epoch {epoch:04d}/{EPOCHS} | "
            f"train {train_loss:.6f} | val {val_loss:.6f} | best {best_val:.6f}"
        )

if best_state is not None:
    model.load_state_dict(best_state)

checkpoint = {
    "model_state": model.state_dict(),
    "action_mean": torch.tensor(dataset.action_mean, dtype=torch.float32),
    "action_std": torch.tensor(dataset.action_std, dtype=torch.float32),
    "config": {
        "input_dim": INPUT_DIM,
        "action_dim": ACTION_DIM,
        "h_window": H_WINDOW,
        "phase_dim": PHASE_DIM,
        "embed_dim": EMBED_DIM,
        "nhead": NHEAD,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "max_tokens": max(2000, dataset.max_demo_len + 1),
        "model_class": "IntentPromptPolicy2D",
        "feature_mode": "intent_relative",
        "raw_point_dim": RAW_POINT_DIM,
        "intent_feature_dim": INTENT_FEATURE_DIM,
    },
    "training": {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "best_val_loss": best_val,
        "samples": len(dataset),
        "trials": len(raw_data),
    },
}

torch.save(checkpoint, MODEL_OUT)
print(f"Saved: {MODEL_OUT.resolve()}")

if IN_COLAB:
    files.download(str(MODEL_OUT))
