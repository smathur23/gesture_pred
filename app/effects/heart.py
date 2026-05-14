from __future__ import annotations

import math
import random

import cv2
import numpy as np

from app.effects.base import BaseEffect, bbox_center, draw_text_badge


class HeartEffect(BaseEffect):
    def __init__(self) -> None:
        super().__init__(duration_sec=1.4)
        self.hearts: list[tuple[float, float, float, float]] = []

    def on_trigger(self) -> None:
        self.hearts = []
        if not self.last_trigger:
            return
        cx, cy = bbox_center(self.last_trigger.detection, self.last_trigger.frame_shape)
        for _ in range(22):
            self.hearts.append(
                (
                    float(cx + random.uniform(-40, 40)),
                    float(cy + random.uniform(-10, 10)),
                    random.uniform(-30, 30),
                    random.uniform(-120, -30),
                )
            )

    def render(self, frame: np.ndarray, elapsed: float) -> np.ndarray:
        draw_text_badge(frame, "HEART", (20, 60), bg_color=(70, 20, 90), fg_color=(255, 255, 255))
        dt = 1.0 / 30.0
        next_hearts: list[tuple[float, float, float, float]] = []
        for x, y, vx, vy in self.hearts:
            x += vx * dt + math.sin(elapsed * 7.0) * 0.5
            y += vy * dt
            vy += 30 * dt
            if y > -20:
                self._draw_heart(frame, int(x), int(y), 8, (180, 80, 255))
                next_hearts.append((x, y, vx, vy))
        self.hearts = next_hearts
        return frame

    @staticmethod
    def _draw_heart(frame: np.ndarray, x: int, y: int, size: int, color: tuple[int, int, int]) -> None:
        cv2.circle(frame, (x - size // 2, y - size // 3), size // 2, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x + size // 2, y - size // 3), size // 2, color, -1, cv2.LINE_AA)
        pts = np.array(
            [
                (x - size, y - size // 4),
                (x + size, y - size // 4),
                (x, y + size),
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(frame, pts, color, cv2.LINE_AA)

