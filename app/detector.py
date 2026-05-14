from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]


class GestureDetector:
    def __init__(
        self,
        model_path: str,
        class_id_to_name: Dict[int, str],
        per_class_thresholds: Dict[str, float],
        default_threshold: float = 0.4,
        device: str | int = "cpu",
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics/YOLO is required. Install YOLOv13 package first "
                "(e.g., pip install -e vendor/yolov13)."
            ) from exc

        self.model = YOLO(model_path)
        self.class_id_to_name = class_id_to_name
        self.per_class_thresholds = per_class_thresholds
        self.default_threshold = default_threshold
        self.device = device

    def infer(self, frame: np.ndarray, imgsz: int = 640) -> List[Detection]:
        results = self.model.predict(
            source=frame,
            imgsz=imgsz,
            verbose=False,
            device=self.device,
            conf=0.01,
        )
        if not results:
            return []
        return self._parse_result(results[0])

    def _parse_result(self, result) -> List[Detection]:
        detections: List[Detection] = []
        boxes = result.boxes
        if boxes is None:
            return detections
        for box in boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            class_name = self.class_id_to_name.get(class_id, str(class_id))
            threshold = self.per_class_thresholds.get(class_name, self.default_threshold)
            if confidence < threshold:
                continue
            xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox_xyxy=xyxy,  # type: ignore[arg-type]
                )
            )
        detections.sort(key=lambda det: det.confidence, reverse=True)
        return detections

    @staticmethod
    def dominant_detection(detections: List[Detection]) -> Optional[Detection]:
        if not detections:
            return None
        return detections[0]

