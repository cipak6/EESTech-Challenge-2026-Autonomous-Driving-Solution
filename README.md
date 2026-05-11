# EESTech Challenge 2026 — Autonomous Driving Agent

**4th place** at the EESTech Challenge 2026 autonomous driving competition hosted on the [MetaDrive](https://github.com/metadriverse/metadrive) simulator.

Built by **Sergej Čipak**[cipak6](https://github.com/cipak6) and **Marko Vukmirović** [@MarkoVukmirovic02](https://github.com/MarkoVukmirovic02).

---

## Overview

The task was to build an agent that drives autonomously in MetaDrive — a procedurally generated traffic simulator — across maps with straight roads, roundabouts, and curves, in the presence of traffic.

Our solution is a **hybrid agent** combining three components:

1. **Behavioural Cloning (BC) neural network** — learns steering from expert demonstrations
2. **Rule-based teacher controller** — handles throttle, lane keeping, and navigation
3. **YOLO + Kalman perception pipeline** — detects and tracks surrounding vehicles, predicts their future positions, and informs braking decisions

---

## Architecture

```
Simulator observation (259-dim state vector)
        │
        ├──► DrivingMLP (BC model) ──────────────────► steering
        │         ↑
        │    trained on expert data
        │
        └──► TeacherSolution (rule-based) ──────────► throttle
                    ↑
             lidar + nav commands
                    │
             PerceptionRiskModule
                    ├── YOLOv8 detection + ByteTrack
                    ├── Kalman filter per tracked object
                    └── IoU-based risk scoring
                              │
                         danger flag ──► throttle override
```

### Behavioural Cloning

The BC model (`DrivingMLP`) is a 3-layer MLP (256 → 256 → 128 → 1) trained to predict steering from the 259-dimensional state observation. During inference, lidar channels (indices 19–258) are masked to 1.0 so the model generalises purely from road geometry (lane boundaries, heading, navigation checkpoints).

Training uses **weighted Huber loss** — samples from human steering corrections are weighted 3× higher than controller demonstrations to prioritise recovery behaviour. The data pipeline splits by episode rather than by sample to prevent temporal leakage between train and test sets.

### Rule-Based Teacher

`TeacherSolution` handles all throttle decisions and serves as the label source during data collection. It uses:

- **Lane keeping** — PD controller on boundary error with yaw-rate damping
- **Lidar obstacle avoidance** — 5 angular sectors (front, front-left, front-right, left, right) with hard danger/caution thresholds
- **Road-edge recovery** — stateful recovery mode when either boundary drops below a threshold
- **Navigation following** — left/right steering bias on navigation commands

### Perception & Risk Engine

The perception pipeline runs adaptively: it operates at a low idle rate (every 60 frames) and switches to a high active rate (every 5 frames) when the lidar signature indicates a nearby object. This keeps CPU load low on open road while staying responsive near traffic.

Per tracked object, a **BoxKalmanFilter** (8-state: cx, cy, w, h, vx, vy, vw, vh) predicts where the bounding box will be in the next frame. The **risk engine** scores four candidate actions (go, slow\_down, brake, stop) by computing IoU between each action's predicted ego corridor and the predicted object boxes. The safest action is passed to the teacher controller as a throttle override.

---

## Data Collection Modes

`game.py` supports multiple data collection modes configurable via `DATA_MODE`:

| Mode | Drives | Labels |
|---|---|---|
| `model` | BC model | BC model |
| `controller` | Teacher | Teacher |
| `dagger` | BC model | Teacher (DAgger) |
| `human` | Human keyboard | Human keyboard |
| `human_assist` | BC model / human (hold SHIFT) | Teacher / human correction |

`human_assist` was the most useful during development: the model drives normally, but holding SHIFT lets you correct steering while the teacher still provides reliable throttle. These corrections are saved with 3× weight.

---

## File Structure

```
├── main.py                     # Entry point
├── game.py                     # Simulation loop, data collection, adaptive perception
├── solution.py                 # Hybrid agent (BC steering + teacher throttle)
├── teacher_solution.py         # Rule-based controller
├── perception_risk_module.py   # Kalman tracking + risk orchestration
├── risk_engine.py              # IoU-based action risk scoring
├── yolov8n.py                  # YOLOv8 + ByteTrack wrapper
├── control.py                  # Keyboard input with ramping/decay
├── logger.py                   # ActionLogger + DatasetWriter
├── training/
│   └── train_bc.py             # BC training script
├── models/
│   ├── 1.pt                    # Final BC model checkpoint (used in solution.py)
│   ├── bc_mlp_bestBCP.pt       # Earlier BC checkpoint
│   ├── bc_steer_only.pt        # Steering-only ablation checkpoint
│   └── yolo_carla_best.pt      # YOLOv8 fine-tuned on CARLA data
└── datasets/                   # Collected driving data (gitignored)
```

---

## Running

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the agent:**
```bash
python main.py
```

**Collect training data** (set `DATA_MODE` in `game.py` CONFIG):
```bash
# Edit CONFIG["DATA_MODE"] = "human_assist" then:
python main.py
```

**Train the BC model:**
```bash
python -m training.train_bc --dataset-dir datasets/ --output models/bc_mlp_best.pt
```

---

## Controls

| Key | Action |
|---|---|
| A / D | Steer left / right |
| W | Accelerate |
| S | Brake |
| LSHIFT | Human override (in `human_assist` mode) |
| Q / ESC | Quit and save log |
