from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from app.effects.base import BaseEffect, bbox_center, draw_text_badge, random_color


@dataclass
class Spark:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    color: tuple[int, int, int]


class ThumbsUpEffect(BaseEffect):
    def __init__(self) -> None:
        super().__init__(duration_sec=1.1)
        self.sparks: List[Spark] = []

    def on_trigger(self) -> None:
        self.sparks = []
        if not self.last_trigger:
            return
        cx, cy = bbox_center(self.last_trigger.detection, self.last_trigger.frame_shape)
        for _ in range(45):
            self.sparks.append(
                Spark(
                    x=float(cx),
                    y=float(cy),
                    vx=random.uniform(-160, 160),
                    vy=random.uniform(-260, -20),
                    life=random.uniform(0.35, 1.0),
                    color=random_color(bright=True),
                )
            )

    def render(self, frame: np.ndarray, elapsed: float) -> np.ndarray:
        draw_text_badge(frame, "LIKE", (20, 60), bg_color=(30, 120, 30), fg_color=(255, 255, 255))
        dt = 1.0 / 30.0
        for s in self.sparks:
            if s.life <= 0:
                continue
            s.x += s.vx * dt
            s.y += s.vy * dt
            s.vy += 380 * dt
            s.life -= dt
            r = max(1, int(4 * s.life))
            cv2.circle(frame, (int(s.x), int(s.y)), r, s.color, -1, cv2.LINE_AA)
        return frame

