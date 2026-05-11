"""
Perception + Risk Module
========================
Combines YOLOv8 object detection, Kalman-filter-based tracking, and the IoU
risk engine into a single per-frame interface used by the game loop.
"""

import numpy as np

from yolov8n import YOLOPerception
from risk_engine import compute_action_risk, compute_ego_danger, box_iou


class BoxKalmanFilter:
    """
    8-state Kalman filter for a single bounding box: (cx, cy, w, h, vx, vy, vw, vh).
    Predicts future box positions for risk assessment.
    """

    def __init__(self, cx, cy, w, h, vx, vy, vw, vh):
        self.x = np.array([[cx], [cy], [w], [h], [vx], [vy], [vw], [vh]], dtype=np.float32)
        dt = 1.0

        # State transition: constant-velocity model
        self.F = np.array([
            [1, 0, 0, 0, dt, 0,  0,  0],
            [0, 1, 0, 0, 0,  dt, 0,  0],
            [0, 0, 1, 0, 0,  0,  dt, 0],
            [0, 0, 0, 1, 0,  0,  0,  dt],
            [0, 0, 0, 0, 1,  0,  0,  0],
            [0, 0, 0, 0, 0,  1,  0,  0],
            [0, 0, 0, 0, 0,  0,  1,  0],
            [0, 0, 0, 0, 0,  0,  0,  1],
        ], dtype=np.float32)

        # Observation: we only measure position+size, not velocity
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=np.float32)

        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.Q = np.eye(8, dtype=np.float32) * 0.01
        self.R = np.eye(4, dtype=np.float32) * 5.0
        self.I = np.eye(8, dtype=np.float32)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        z = np.array(z, dtype=np.float32).reshape(4, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P
        return self.x

    def predict_k_steps(self, k):
        future_x = self.x.copy()
        for _ in range(k):
            future_x = self.F @ future_x
        return future_x


class PerceptionRiskModule:
    """
    Per-frame pipeline:
      1. Detect and track objects with YOLOv8.
      2. Maintain per-track Kalman filters and predict future positions.
      3. Score candidate actions via IoU overlap with predicted boxes.
      4. Check whether any predicted box enters the ego danger zone.

    Also tracks a running IoU score between past predictions and actual
    detections, which is useful for evaluating Kalman filter quality.
    """

    def __init__(
        self,
        model_path="models/yolo_carla_best.pt",
        conf=0.10,
        imgsz=640,
        predict_ahead=1,
        min_history=5,
        max_history=10,
    ):
        self.perception = YOLOPerception(
            model_path=model_path,
            conf=conf,
            imgsz=imgsz,
            allowed_classes={"person", "bicycle", "car", "motorcycle", "bus", "truck"},
        )

        self.trackable_classes = {"car", "truck", "bus", "motorcycle", "bicycle", "person"}

        self.track_history = {}
        self.kalman_filters = {}
        self.pending_predictions = {}
        self.iou_scores = []

        self.frame_idx = 0
        self.predict_ahead = predict_ahead
        self.min_history = min_history
        self.max_history = max_history

    def update_tracks_and_predict(self, detections):
        """Update Kalman filters for each tracked object and attach a predicted future box."""
        target_frame = self.frame_idx + self.predict_ahead

        for det in detections:
            if det["class"] not in self.trackable_classes:
                continue

            track_id = det.get("track_id")
            if track_id is None:
                continue

            x1, y1, x2, y2 = det["bbox"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1

            history = self.track_history.setdefault(track_id, [])
            history.append((cx, cy, w, h))
            if len(history) > self.max_history:
                self.track_history[track_id] = history[-self.max_history:]

            if len(history) < self.min_history:
                continue

            if track_id not in self.kalman_filters:
                cx1, cy1, w1, h1 = history[0]
                cxN, cyN, wN, hN = history[-1]
                n = len(history) - 1
                self.kalman_filters[track_id] = BoxKalmanFilter(
                    cxN, cyN, wN, hN,
                    (cxN - cx1) / n, (cyN - cy1) / n,
                    (wN - w1) / n, (hN - h1) / n,
                )

            kf = self.kalman_filters[track_id]
            kf.predict()
            kf.update([cx, cy, w, h])

            fs = kf.predict_k_steps(self.predict_ahead)
            fcx, fcy, fw, fh = fs[0, 0], fs[1, 0], fs[2, 0], fs[3, 0]

            pred_box = [
                int(fcx - fw / 2),
                int(fcy - fh / 2),
                int(fcx + fw / 2),
                int(fcy + fh / 2),
            ]
            det["pred_box"] = pred_box

            self.pending_predictions.setdefault(target_frame, []).append({
                "track_id": track_id,
                "pred_box": pred_box,
                "source_frame": self.frame_idx,
            })

        return detections

    def evaluate_old_predictions(self, detections):
        """Compare stored predictions for this frame against actual detections and record IoU."""
        if self.frame_idx not in self.pending_predictions:
            return

        for prediction in self.pending_predictions[self.frame_idx]:
            for det in detections:
                if det.get("track_id") == prediction["track_id"]:
                    self.iou_scores.append(box_iou(prediction["pred_box"], det["bbox"]))
                    break

        del self.pending_predictions[self.frame_idx]

    def process_frame(self, frame):
        """
        Run the full perception + risk pipeline on a single BGR frame.

        Returns a dict with:
            best_action, action_risks, detections, avg_prediction_iou,
            ego_box, max_ego_iou, danger, danger_threshold, danger_objects
        """
        detections, results = self.perception.detect_and_track(frame)

        self.evaluate_old_predictions(detections)
        detections = self.update_tracks_and_predict(detections)

        frame_h, frame_w = frame.shape[:2]
        best_action, action_risks = compute_action_risk(detections, frame_w, frame_h)
        danger_data = compute_ego_danger(detections, frame_w, frame_h)

        self.frame_idx += 1

        avg_iou = sum(self.iou_scores) / len(self.iou_scores) if self.iou_scores else None

        return {
            "best_action": best_action,
            "action_risks": action_risks,
            "detections": detections,
            "results": results,
            "avg_prediction_iou": avg_iou,
            **danger_data,
        }
