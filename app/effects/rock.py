from __future__ import annotations

import math

import cv2
import numpy as np

from app.effects.base import BaseEffect, bbox_center, draw_text_badge


class RockEffect(BaseEffect):
    def __init__(self) -> None:
        super().__init__(duration_sec=0.7)

    def render(self, frame: np.ndarray, elapsed: float) -> np.ndarray:
        if not self.last_trigger:
            return frame
        h, w = frame.shape[:2]
        cx, cy = bbox_center(self.last_trigger.detection, self.last_trigger.frame_shape)
        offset = int(math.sin(elapsed * 45) * 8)
        shifted = np.roll(frame, shift=offset, axis=1)
        frame[:] = shifted
        draw_text_badge(frame, "ROCK", (20, 60), bg_color=(20, 20, 120), fg_color=(255, 255, 255))
        points = [
            (cx, cy - 40),
            (cx - 15, cy - 5),
            (cx + 4, cy - 5),
            (cx - 20, cy + 36),
            (cx + 22, cy + 2),
            (cx + 5, cy + 2),
            (cx + 20, cy - 35),
        ]
        cv2.polylines(frame, [np.array(points, dtype=np.int32)], False, (255, 255, 90), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (80, 80, 240), 2)
        return frame

