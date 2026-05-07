# vvaw

`vvaw` is the Windows-focused standalone workspace for the current robot visual imitation project.

The core idea is:

```text
current robot camera view + visual demo video -> robot joint actions
```

Right now the most important runnable path is `v2/infer.py`. If you want to use the project on Windows, start there.

## Windows Quick Start For `v2/infer.py`

Open PowerShell in `C:\vvaw`.

### 1. Create the virtual environment

```powershell
python -m venv .venv
```

If `python` is not on PATH, use your installed Python directly, for example:

```powershell
py -3.12 -m venv .venv
```

### 2. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation scripts in the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

After activation, your prompt should show `(.venv)`.

### 3. Install requirements

```powershell
pip install -r requirements.txt
```

This project uses `pybullet-arm64` as a drop-in replacement for `pybullet` on Windows so that PyBullet installs from wheels instead of requiring Microsoft C++ Build Tools.

### 4. Optional: enable NVIDIA GPU for PyTorch

The default `pip install -r requirements.txt` may give you CPU PyTorch depending on the machine and index state. If you want CUDA inference on an NVIDIA GPU, run:

```powershell
pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Then verify CUDA.

From the repo root:

```powershell
python .\check_gpu.py
```

From inside `C:\vvaw\v2`:

```powershell
python ..\check_gpu.py
```

Good output should say:

```text
cuda available: True
active device: 0 - NVIDIA ...
```

### 5. Run `v2/infer.py`

From the repo root:

```powershell
python .\v2\infer.py --demo 5 --phase-speed 0.75 --physics-steps 10
```

Or from inside `C:\vvaw\v2`:

```powershell
python .\infer.py --demo 5 --phase-speed 0.75 --physics-steps 10
```

`--demo 5` means `data/demovideos/demovid5.mp4`.

Use demos `1` through `9` for trained examples. `demovid10.mp4` exists, but it does not have paired training data.

### 6. Force CUDA and fail fast if GPU is missing

```powershell
python .\v2\infer.py --demo 5 --require-cuda
```

That prints the Python path, Torch build, CUDA runtime, and chosen device at startup.

### 7. Make `nvidia-smi` clearly show active GPU memory

```powershell
python .\v2\infer.py --demo 5 --require-cuda --gpu-sentinel-mb 512
```

Then in another terminal:

```cmd
nvidia-smi -l 1
```

## Helper Scripts On Windows

If you prefer helper scripts instead of doing the manual setup yourself:

### Setup

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

With CUDA PyTorch:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1 -Cuda
```

### Smoke test

```powershell
powershell -ExecutionPolicy Bypass -File .\run_smoke_windows.ps1
```

### Run v2 inference

```powershell
powershell -ExecutionPolicy Bypass -File .\run_v2_windows.ps1 -Demo 5
```

### Headless test run

```powershell
powershell -ExecutionPolicy Bypass -File .\run_v2_windows.ps1 -Demo 1 -Headless -MaxSteps 1 -PhysicsSteps 1 -PhaseSpeed 1.0
```

### GPU check

```powershell
powershell -ExecutionPolicy Bypass -File .\check_gpu_windows.ps1
```

## What The Project Is Trying To Do

`vvaw` is an imitation-learning workspace for a robot arm. A demonstration video is treated as a visual prompt. The system observes the robot's own camera view, compares it against the prompt, and predicts actions that make the robot imitate the demonstrated motion.

At the moment the project is focused on a single target object class named `Target`.

## Project Layout

```text
C:\vvaw\
  assets\urdf\       robot model
  data\              demo videos, camview videos, and joint logs
  subsystems\        reusable sim and perception tools
  v1\                older 3D-feature approach
  v2\                main current inference path
  v3\                newer delta-action training experiment
  yolo\              detector weights, datasets, and scripts
  extract_points.py  root 3D dataset builder
  project_paths.py   shared paths and local runtime env defaults
  requirements.txt   Python dependencies
  teleop.ino         ESP32 teleop sketch
```

## Data Layout

The paired data lives in:

```text
data\
  demovideos\
  camview\
  jointdata\
```

Naming is trial-based:

```text
data\demovideos\demovidN.mp4
data\camview\camviewN.mp4
data\jointdata\jdN.csv
```

Trials `1` through `9` are the complete paired examples used for learning.

The joint CSV format is:

```csv
timestamp,j0,j1,j2,j3,j4,j5,d0,d1,d2,d3,d4,d5
```

`j0` through `j5` are target joint positions. `d0` through `d5` are frame-to-frame deltas.

## YOLO Point Format

The detector uses 7 classes:

```text
0 J0_Base
1 J1_Shoulder
2 J2_Elbow
3 J3_Wrist
4 J4_GripL
5 J5_GripR
6 Target
```

The v2 and v3 2D representation stores:

```text
J0 x,y
J1 x,y
J2 x,y
J3 x,y
J4 x,y
J5 x,y
Target x,y
```

That is 14 values per frame.

## `v2` Overview

`v2` is the current single-object intent-relative approach.

Files:

```text
v2\dataset_2d.npy
v2\extract_points_2d.py
v2\colab_train_intent_single_cell.py
v2\infer.py
v2\model2_vla_2d_intent.pth
```

The important modeling idea in `v2` is that raw YOLO points are transformed into intent-relative geometry before being fed to the policy. That includes:

```text
arm shape relative to the base
end-effector relative position
gripper width
target direction
target distance
```

`v2/infer.py` is the main inference entry point.

Useful flags:

```text
--demo N
--phase-speed X
--physics-steps N
--require-cuda
--gpu-sentinel-mb N
--direct
--no-camera-window
--max-steps N
```

## `v1` Overview

`v1` is the older 3D-feature approach.

Files:

```text
v1\train.py
v1\train1.py
v1\infer.py
v1\model1_vla.pth
dataset.npy
```

It uses a 22-feature representation and predicts six joint actions.

## `v3` Overview

`v3` is the newer delta-action experiment.

Files:

```text
v3\dataset_2d.npy
v3\train_v3_delta_intent_single_cell.ipynb
```

The main change from `v2` is that `v3` predicts joint deltas instead of absolute joint targets, with separate output heads for arm and gripper behavior.

There is not yet a finished `v3/infer.py`.

## Subsystems

`subsystems\` contains reusable utilities:

```text
combinedinferlite.py
combinedinferlite2.py
testsimulation.py
```

`subsystems/testsimulation.py` is the basic PyBullet smoke test.

Run it:

```powershell
python .\subsystems\testsimulation.py
```

Headless:

```powershell
python .\subsystems\testsimulation.py --direct --steps 120 --no-camera-window
```

## Dataset Builders

Build the root 3D dataset:

```powershell
python .\extract_points.py
```

Build the `v2` 2D dataset:

```powershell
python .\v2\extract_points_2d.py
```

## Teleop Controller

`teleop.ino` is the ESP32 sketch for the physical teleop arm.

It reads:

```text
yaw       GPIO 36
shoulder  GPIO 39
elbow     GPIO 34
end link  GPIO 35
button    GPIO 12
```

After a 5-second calibration period it streams:

```text
yaw,shoulder,elbow,endlink,button
```

The PC-side code expects this 5-value serial format at `115200` baud.

## Recommended First Commands

If you are coming back later and just want the shortest reliable Windows path:

```powershell
cd C:\vvaw
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
python .\v2\infer.py --demo 5 --require-cuda
```

## Notes

Older helper files mentioned in older conversations such as `infer2.py` or `intent_features.py` are obsolete. The current `v2` inference path is consolidated into `v2/infer.py`.
