from ultralytics import YOLO


class YOLOPerception:
    """
    Thin wrapper around a YOLOv8 model with ByteTrack tracking enabled.
    Filters detections to a configurable set of allowed classes.
    """

    def __init__(self, model_path, conf=0.25, imgsz=800, allowed_classes=None):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.allowed_classes = allowed_classes

    def detect_and_track(self, frame):
        """
        Run detection + tracking on a single BGR frame.

        Returns:
            detections (list[dict]): each with keys track_id, class, confidence, bbox
            results: raw Ultralytics Results object
        """
        results = self.model.track(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            persist=True,
            verbose=False,
        )[0]

        detections = []
        if results.boxes is None:
            return detections, results

        for box in results.boxes:
            cls_name = self.model.names[int(box.cls[0])]
            if self.allowed_classes is not None and cls_name not in self.allowed_classes:
                continue

            detections.append({
                "track_id": int(box.id[0]) if box.id is not None else None,
                "class": cls_name,
                "confidence": float(box.conf[0]),
                "bbox": tuple(map(int, box.xyxy[0])),
            })

        return detections, results
