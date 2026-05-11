# ---------------------------------------------------------------------------
# Risk engine — scores candidate longitudinal actions by predicted overlap
# with detected objects, using a simple IoU-based metric.
# ---------------------------------------------------------------------------

driving_actions = ["go", "slow_down", "brake", "stop"]

# IoU thresholds
MIN_IOU = 0.03      # below this, an object is ignored for risk scoring
STOP_IOU = 0.25     # above this on any action, force a stop

DEFAULT_DANGER_IOU = 0.05   # threshold for ego-zone danger alert


def make_ego_corridor(frame_w, frame_h):
    """
    Static danger zone in front of the ego vehicle, in image pixel coordinates.
    Covers roughly the centre third of the frame width and the lower half.
    """
    return [
        int(frame_w * 0.35),
        int(frame_h * 0.45),
        int(frame_w * 0.65),
        int(frame_h * 0.95),
    ]


def predict_ego_corridor(action, frame_w, frame_h, ego_speed_px_per_frame=8, horizon=5):
    """
    Approximate where the ego corridor will extend over the next `horizon` frames
    under a candidate longitudinal action.
    """
    x1, y1, x2, y2 = make_ego_corridor(frame_w, frame_h)

    if action == "go":
        shift = ego_speed_px_per_frame * horizon
    elif action == "slow_down":
        shift = ego_speed_px_per_frame * horizon * 0.4
    else:
        shift = 0

    return [x1, max(0, y1 - int(shift)), x2, max(0, y2 - int(shift))]


def box_iou(box_a, box_b):
    """Intersection-over-Union for two axis-aligned bounding boxes [x1,y1,x2,y2]."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def compute_action_risk(detections, frame_w, frame_h):
    """
    Score each candidate action by the maximum predicted-box overlap with the
    action's ego corridor. Objects in the top 45% of the frame are ignored
    (they are still far away).

    Returns:
        best_action (str): safest candidate action
        action_risks (dict[str, float]): per-action risk scores
    """
    action_risks = {}
    max_overlap = 0.0

    for action in driving_actions:
        ego_future = predict_ego_corridor(action, frame_w, frame_h)
        max_risk = 0.0

        for det in detections:
            pred_box = det.get("pred_box")
            if pred_box is None:
                continue

            cy = (pred_box[1] + pred_box[3]) / 2
            if cy < frame_h * 0.45:
                continue

            iou = box_iou(ego_future, pred_box)
            max_overlap = max(max_overlap, iou)

            if iou < MIN_IOU:
                continue

            # Objects lower in the image are closer → higher urgency weight.
            risk = iou * (cy / frame_h)
            max_risk = max(max_risk, risk)

        action_risks[action] = max_risk

    if max_overlap >= STOP_IOU:
        return "stop", action_risks

    best_action = min(action_risks, key=action_risks.get)
    return best_action, action_risks


def compute_ego_danger(detections, frame_w, frame_h, danger_threshold=DEFAULT_DANGER_IOU):
    """
    Check whether any predicted object box overlaps the static ego danger zone.

    Complements compute_action_risk():
      - compute_action_risk()  → which action looks safest?
      - compute_ego_danger()   → is anything predicted to enter our danger zone?
    """
    ego_box = make_ego_corridor(frame_w, frame_h)
    max_ego_iou = 0.0
    danger_objects = []

    for det in detections:
        pred_box = det.get("pred_box")
        if pred_box is None:
            continue

        cy = (pred_box[1] + pred_box[3]) / 2
        if cy < frame_h * 0.45:
            continue

        ego_iou = box_iou(ego_box, pred_box)
        max_ego_iou = max(max_ego_iou, ego_iou)

        if ego_iou >= danger_threshold:
            danger_objects.append({
                "track_id": det.get("track_id"),
                "class": det.get("class"),
                "confidence": det.get("confidence"),
                "bbox": det.get("bbox"),
                "pred_box": pred_box,
                "ego_iou": ego_iou,
            })

    return {
        "ego_box": ego_box,
        "max_ego_iou": max_ego_iou,
        "danger": len(danger_objects) > 0,
        "danger_threshold": danger_threshold,
        "danger_objects": danger_objects,
    }
