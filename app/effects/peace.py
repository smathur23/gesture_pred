from __future__ import annotations

import random

import cv2
import numpy as np

from app.effects.base import BaseEffect, bbox_center, draw_text_badge, random_color


class PeaceEffect(BaseEffect):
    def __init__(self) -> None:
        super().__init__(duration_sec=1.1)
        self.pieces: list[tuple[float, float, float, float, tuple[int, int, int]]] = []

    def on_trigger(self) -> None:
        self.pieces = []
        if not self.last_trigger:
            return
        cx, cy = bbox_center(self.last_trigger.detection, self.last_trigger.frame_shape)
        for _ in range(55):
            self.pieces.append(
                (
                    float(cx),
                    float(cy),
                    random.uniform(-180, 180),
                    random.uniform(-220, -40),
                    random_color(bright=True),
                )
            )

    def render(self, frame: np.ndarray, elapsed: float) -> np.ndarray:
        draw_text_badge(frame, "PEACE", (20, 60), bg_color=(40, 40, 140), fg_color=(255, 255, 255))
        dt = 1.0 / 30.0
        next_pieces: list[tuple[float, float, float, float, tuple[int, int, int]]] = []
        for x, y, vx, vy, color in self.pieces:
            x += vx * dt
            y += vy * dt
            vy += 360 * dt
            if y < frame.shape[0] + 20:
                cv2.rectangle(frame, (int(x), int(y)), (int(x) + 4, int(y) + 10), color, -1)
                next_pieces.append((x, y, vx, vy, color))
        self.pieces = next_pieces
        return frame

