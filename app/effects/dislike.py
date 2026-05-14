from __future__ import annotations

import random

import cv2
import numpy as np

from app.effects.base import BaseEffect, draw_text_badge


class DislikeEffect(BaseEffect):
    def __init__(self) -> None:
        super().__init__(duration_sec=1.2)
        self.drops: list[tuple[float, float, float]] = []

    def on_trigger(self) -> None:
        self.drops = [(random.uniform(0, 1), random.uniform(-1, 0), random.uniform(0.5, 1.4)) for _ in range(90)]

    def render(self, frame: np.ndarray, elapsed: float) -> np.ndarray:
        h, w = frame.shape[:2]
        draw_text_badge(frame, "DISLIKE", (20, 60), bg_color=(120, 30, 30), fg_color=(255, 255, 255))
        for i, (nx, ny, speed) in enumerate(self.drops):
            y = int((ny + elapsed * speed) * h)
            x = int(nx * w)
            y2 = y + 12
            cv2.line(frame, (x, y), (x, y2), (180, 180, 255), 1, cv2.LINE_AA)
            self.drops[i] = (nx, ny, speed)
        cloud_alpha = min(0.35, elapsed * 0.6)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, int(h * 0.22)), (70, 70, 70), -1)
        frame[:] = cv2.addWeighted(overlay, cloud_alpha, frame, 1 - cloud_alpha, 0.0)
        return frame

