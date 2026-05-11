"""
Behavioural Cloning Training Script
=====================================
Trains a DrivingMLP to clone steering from JSONL datasets collected by game.py.

Usage:
    python -m training.train_bc --dataset-dir datasets/ --output models/bc_mlp_best.pt

The script:
  - Loads all .jsonl files from the dataset directory
  - Splits data by episode to prevent leakage between train/val/test
  - Applies per-source sample weights (human corrections weighted highest)
  - Trains with Huber loss and early stopping on validation loss
  - Saves the best checkpoint with normalisation statistics embedded
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DrivingMLP(nn.Module):
    """3-hidden-layer MLP that maps a (normalised) 259-dim state vector to a steering value."""

    def __init__(self, input_dim: int = 259) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),  # output stays in [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DrivingDataset(Dataset):
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        sample_weights: np.ndarray,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
    ) -> None:
        self.observations = observations.astype(np.float32)
        self.actions = actions.astype(np.float32)
        self.sample_weights = sample_weights.astype(np.float32)
        self.obs_mean = obs_mean.astype(np.float32)
        self.obs_std = obs_std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int):
        x = (self.observations[idx] - self.obs_mean) / self.obs_std
        return (
            torch.from_numpy(x),
            torch.from_numpy(self.actions[idx]),
            torch.tensor(self.sample_weights[idx], dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Per-source sample weights used during training
_SOURCE_WEIGHTS = {
    "human": 1.5,
    "human_correction": 3.0,   # explicit human steering correction — highest priority
    "dagger_controller": 1.25,
    "controller": 0.1,
}

_VALID_SOURCES = set(_SOURCE_WEIGHTS.keys())


def load_jsonl_files(dataset_dir: Path, human_weight: float, controller_weight: float) -> List[dict]:
    """
    Load all .jsonl files from dataset_dir. Skips malformed lines.
    human_weight and controller_weight override the defaults for those two sources.
    """
    weights = dict(_SOURCE_WEIGHTS)
    weights["human"] = human_weight
    weights["controller"] = controller_weight

    records: List[dict] = []
    files = sorted(dataset_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in: {dataset_dir}")

    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[SKIP] Invalid JSON: {path.name}:{line_num}")
                    continue

                obs = row.get("observation")
                steer = row.get("action_steering")
                source = row.get("source", "unknown")

                if not isinstance(obs, list) or len(obs) != 259:
                    continue
                if steer is None or source not in _VALID_SOURCES:
                    continue

                try:
                    obs_arr = np.asarray(obs, dtype=np.float32)
                    action_arr = np.asarray([steer], dtype=np.float32)
                except (TypeError, ValueError):
                    continue

                if not np.all(np.isfinite(obs_arr)) or not np.all(np.isfinite(action_arr)):
                    continue

                records.append({
                    "file": path.name,
                    "episode_id": row.get("episode_id", 0),
                    "source": source,
                    "observation": obs_arr,
                    "action": action_arr,
                    "weight": weights[source],
                })

    if not records:
        raise RuntimeError("No valid training records found.")
    return records


def split_by_episode(
    records: List[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Split records by episode rather than by sample to prevent data leakage.
    Falls back to sample-level split when fewer than 3 episodes are found.
    """
    episode_groups: Dict[str, List[dict]] = defaultdict(list)
    for row in records:
        # Include the filename because episode_id resets each run
        key = f"{row['file']}::episode_{row['episode_id']}"
        episode_groups[key].append(row)

    episode_keys = list(episode_groups.keys())
    random.Random(seed).shuffle(episode_keys)

    if len(episode_keys) < 3:
        print("[WARN] Fewer than 3 episodes found; falling back to sample-level split.")
        shuffled = records[:]
        random.Random(seed).shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]

    n_total = len(episode_keys)
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))

    train_keys = set(episode_keys[:n_train])
    val_keys = set(episode_keys[n_train:n_train + n_val])

    train_rows, val_rows, test_rows = [], [], []
    for key, rows in episode_groups.items():
        if key in train_keys:
            train_rows.extend(rows)
        elif key in val_keys:
            val_rows.extend(rows)
        else:
            test_rows.extend(rows)

    return train_rows, val_rows, test_rows


def rows_to_arrays(rows: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.stack([r["observation"] for r in rows]),
        np.stack([r["action"] for r in rows]),
        np.asarray([r["weight"] for r in rows], dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_steer_abs = 0.0
    total_count = 0

    for x, y, w in loader:
        x, y, w = x.to(device), y.to(device), w.to(device)

        with torch.set_grad_enabled(training):
            pred = model(x)
            loss_per_sample = criterion(pred, y).mean(dim=1)
            loss = (loss_per_sample * w).sum() / w.sum()

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_steer_abs += torch.abs(pred - y)[:, 0].sum().item()
        total_count += batch_size

    n = max(total_count, 1)
    return total_loss / n, total_steer_abs / n


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train DrivingMLP via behavioural cloning.")
    parser.add_argument("--dataset-dir",       type=str,   default="datasets")
    parser.add_argument("--output",            type=str,   default="models/bc_mlp_best.pt")
    parser.add_argument("--epochs",            type=int,   default=80)
    parser.add_argument("--batch-size",        type=int,   default=256)
    parser.add_argument("--lr",                type=float, default=1e-3)
    parser.add_argument("--weight-decay",      type=float, default=1e-4)
    parser.add_argument("--human-weight",      type=float, default=1.5)
    parser.add_argument("--controller-weight", type=float, default=1.0)
    parser.add_argument("--train-ratio",       type=float, default=0.8)
    parser.add_argument("--val-ratio",         type=float, default=0.1)
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--patience",          type=int,   default=12)
    args = parser.parse_args()

    set_seed(args.seed)
    dataset_dir = Path(args.dataset_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("[LOAD] Reading dataset …")
    records = load_jsonl_files(dataset_dir, args.human_weight, args.controller_weight)

    source_counts = defaultdict(int)
    for r in records:
        source_counts[r["source"]] += 1
    print(f"[LOAD] Total samples: {len(records)}")
    for src, count in sorted(source_counts.items()):
        print(f"       {src}: {count}")

    train_rows, val_rows, test_rows = split_by_episode(
        records, args.train_ratio, args.val_ratio, args.seed,
    )
    if not train_rows or not val_rows or not test_rows:
        raise RuntimeError("Train/val/test split produced an empty set. Collect more episodes.")

    x_train, y_train, w_train = rows_to_arrays(train_rows)
    x_val,   y_val,   w_val   = rows_to_arrays(val_rows)
    x_test,  y_test,  w_test  = rows_to_arrays(test_rows)

    obs_mean = x_train.mean(axis=0)
    obs_std  = x_train.std(axis=0)
    obs_std[obs_std < 1e-6] = 1.0

    print(f"[SPLIT] Train: {len(train_rows)}  Val: {len(val_rows)}  Test: {len(test_rows)}")

    def make_loader(x, y, w, shuffle):
        ds = DrivingDataset(x, y, w, obs_mean, obs_std)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(x_train, y_train, w_train, shuffle=True)
    val_loader   = make_loader(x_val,   y_val,   w_val,   shuffle=False)
    test_loader  = make_loader(x_test,  y_test,  w_test,  shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    model = DrivingMLP(input_dim=259).to(device)
    criterion = nn.HuberLoss(delta=1.0, reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mae = run_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_mae   = run_epoch(model, val_loader,   None,      criterion, device)

        print(
            f"[EPOCH {epoch:03d}]  "
            f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  val_steer_mae={val_mae:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_left = args.patience
            torch.save({
                "model_state_dict":   model.state_dict(),
                "obs_mean":           obs_mean,
                "obs_std":            obs_std,
                "input_dim":          259,
                "human_weight":       args.human_weight,
                "controller_weight":  args.controller_weight,
            }, output_path)
            print(f"  [SAVE] Best checkpoint → {output_path}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("[EARLY STOP] Validation loss did not improve.")
                break

    print("[TEST] Loading best checkpoint …")
    checkpoint = torch.load(output_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_mae = run_epoch(model, test_loader, None, criterion, device)
    print(f"[TEST] loss={test_loss:.6f}  steer_mae={test_mae:.4f}")


if __name__ == "__main__":
    main()
