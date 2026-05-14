from __future__ import annotations

import random
from dataclasses import dataclass
from time import monotonic
from typing import Optional, Tuple

import cv2
import numpy as np

from app.detector import Detection


@dataclass
class EffectTrigger:
    detection: Optional[Detection]
    frame_shape: Tuple[int, int, int]


class BaseEffect:
    def __init__(self, duration_sec: float = 1.0) -> None:
        self.duration_sec = duration_sec
        self.start_ts: float = 0.0
        self.active: bool = False
        self.last_trigger: Optional[EffectTrigger] = None

    def trigger(self, detection: Optional[Detection], frame_shape: Tuple[int, int, int]) -> None:
        self.start_ts = monotonic()
        self.active = True
        self.last_trigger = EffectTrigger(detection=detection, frame_shape=frame_shape)
        self.on_trigger()

    def on_trigger(self) -> None:
        """Hook for subclasses."""

    def update(self, frame: np.ndarray, dt: float) -> np.ndarray:
        if not self.active:
            return frame
        elapsed = monotonic() - self.start_ts
        if elapsed > self.duration_sec:
            self.active = False
            return frame
        return self.render(frame, elapsed)

    def render(self, frame: np.ndarray, elapsed: float) -> np.ndarray:
        return frame


def bbox_center(detection: Optional[Detection], frame_shape: Tuple[int, int, int]) -> Tuple[int, int]:
    h, w = frame_shape[:2]
    if detection is None:
        return w // 2, h // 2
    x1, y1, x2, y2 = detection.bbox_xyxy
    return int((x1 + x2) * 0.5), int((y1 + y2) * 0.5)


def clip_point(x: int, y: int, frame: np.ndarray) -> Tuple[int, int]:
    h, w = frame.shape[:2]
    return max(0, min(w - 1, x)), max(0, min(h - 1, y))


def alpha_blend(frame: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    alpha = max(0.0, min(1.0, alpha))
    return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0)


def draw_text_badge(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    bg_color: Tuple[int, int, int] = (20, 20, 20),
    fg_color: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(frame, (x - 8, y - th - 10), (x + tw + 8, y + 8), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, scale, fg_color, thickness, cv2.LINE_AA)


def random_color(bright: bool = True) -> Tuple[int, int, int]:
    if bright:
        return (
            random.randint(120, 255),
            random.randint(120, 255),
            random.randint(120, 255),
        )
    return (
        random.randint(30, 150),
        random.randint(30, 150),
        random.randint(30, 150),
    )

