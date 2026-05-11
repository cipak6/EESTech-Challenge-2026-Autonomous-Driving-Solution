import json
import os
import time
from typing import Any


def make_json_safe(value: Any):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if hasattr(value, "tolist"):
        return make_json_safe(value.tolist())
    return str(value)


class ActionLogger:
    """Accumulates per-step log records in memory and writes them to JSON on save()."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.records: list[dict] = []

        folder = os.path.dirname(filepath)
        if folder:
            os.makedirs(folder, exist_ok=True)

        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    self.records = json.load(f)
                print(f"[Logger] Loaded {len(self.records)} existing records from {filepath}")
            except (json.JSONDecodeError, IOError):
                print(f"[Logger] Could not read {filepath}, starting fresh.")

    def log(self, step: int, **kwargs):
        record = {"timestamp": time.time(), "step": step}
        for key, value in kwargs.items():
            record[key] = make_json_safe(value)
        self.records.append(record)

    def save(self):
        try:
            folder = os.path.dirname(self.filepath)
            if folder:
                os.makedirs(folder, exist_ok=True)
            with open(self.filepath, "w") as f:
                json.dump(self.records, f, indent=2)
            print(f"[Logger] Saved {len(self.records)} records -> {self.filepath}")
        except IOError as e:
            print(f"[Logger] Save failed: {e}")

    def summary(self):
        if not self.records:
            print("[Logger] No records to summarize.")
            return

        print("\n" + "=" * 50)
        print("  SESSION SUMMARY")
        print("=" * 50)
        print(f"  Total steps: {len(self.records)}")

        if "action_steering" in self.records[0]:
            steers = [r.get("action_steering", 0.0) for r in self.records]
            print(f"  Action steering:  min={min(steers):.3f}  max={max(steers):.3f}")

        if "action_throttle" in self.records[0]:
            throttles = [r.get("action_throttle", 0.0) for r in self.records]
            print(f"  Action throttle:  min={min(throttles):.3f}  max={max(throttles):.3f}")


class DatasetWriter:
    """Streams training transitions to a JSONL file, flushing periodically."""

    def __init__(self, filepath: str, flush_every: int = 200):
        self.filepath = filepath
        self.flush_every = flush_every
        self.count = 0

        folder = os.path.dirname(filepath)
        if folder:
            os.makedirs(folder, exist_ok=True)

        self._file = open(filepath, "a", encoding="utf-8")

    def log(self, **kwargs):
        record = {"timestamp": time.time()}
        for key, value in kwargs.items():
            record[key] = make_json_safe(value)
        self._file.write(json.dumps(record) + "\n")
        self.count += 1
        if self.count % self.flush_every == 0:
            self._file.flush()

    def close(self):
        if not self._file.closed:
            self._file.flush()
            self._file.close()
            print(f"[Dataset] Saved {self.count} samples -> {self.filepath}")
