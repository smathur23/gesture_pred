from __future__ import annotations

import cv2
import numpy as np

from app.effects.base import BaseEffect, draw_text_badge


class MiddleFingerEffect(BaseEffect):
    def __init__(self) -> None:
        super().__init__(duration_sec=0.9)

    def render(self, frame: np.ndarray, elapsed: float) -> np.ndarray:
        if not self.last_trigger or self.last_trigger.detection is None:
            return frame
        x1, y1, x2, y2 = [int(v) for v in self.last_trigger.detection.bbox_xyxy]
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, frame.shape[1] - 1)
        y2 = min(y2, frame.shape[0] - 1)
        region = frame[y1:y2, x1:x2]
        if region.size > 0:
            small = cv2.resize(region, (max(4, region.shape[1] // 15), max(4, region.shape[0] // 15)))
            censored = cv2.resize(small, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_NEAREST)
            frame[y1:y2, x1:x2] = censored
        if int(elapsed * 10) % 2 == 0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 120), -1)
            frame[:] = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0.0)
        draw_text_badge(frame, "WARNING", (20, 60), bg_color=(20, 20, 120), fg_color=(255, 255, 255))
        return frame

