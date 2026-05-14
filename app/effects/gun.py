from __future__ import annotations

import cv2
import numpy as np

from app.effects.base import BaseEffect, bbox_center, draw_text_badge


class GunEffect(BaseEffect):
    def __init__(self) -> None:
        super().__init__(duration_sec=0.8)

    def render(self, frame: np.ndarray, elapsed: float) -> np.ndarray:
        if not self.last_trigger:
            return frame
        h, w = frame.shape[:2]
        cx, cy = bbox_center(self.last_trigger.detection, self.last_trigger.frame_shape)
        length = int((elapsed / self.duration_sec) * (w * 0.5))
        x2 = min(w - 1, cx + length)
        cv2.line(frame, (cx, cy), (x2, cy), (255, 255, 50), 6, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (x2, cy), (255, 120, 50), 2, cv2.LINE_AA)
        draw_text_badge(frame, "PEW", (20, 60), bg_color=(70, 20, 100), fg_color=(255, 255, 255))
        return frame

