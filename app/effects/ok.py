from __future__ import annotations

import cv2
import numpy as np

from app.effects.base import BaseEffect, bbox_center, draw_text_badge


class OkPulseEffect(BaseEffect):
    def __init__(self) -> None:
        super().__init__(duration_sec=1.1)

    def render(self, frame: np.ndarray, elapsed: float) -> np.ndarray:
        if not self.last_trigger:
            return frame
        cx, cy = bbox_center(self.last_trigger.detection, self.last_trigger.frame_shape)
        draw_text_badge(frame, "OK", (20, 60), bg_color=(20, 100, 120), fg_color=(255, 255, 255))
        for i in range(3):
            phase = (elapsed * 1.8 + i * 0.22) % 1.0
            radius = int(20 + phase * 120)
            alpha = max(0.0, 1.0 - phase)
            color = (int(255 * alpha), 255, 120)
            cv2.circle(frame, (cx, cy), radius, color, 2, cv2.LINE_AA)
        return frame

