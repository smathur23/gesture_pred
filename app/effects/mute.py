from __future__ import annotations

import cv2
import numpy as np

from app.effects.base import BaseEffect, alpha_blend, draw_text_badge


class MuteEffect(BaseEffect):
    def __init__(self) -> None:
        super().__init__(duration_sec=1.4)

    def render(self, frame: np.ndarray, elapsed: float) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = alpha_blend(frame, overlay, alpha=min(0.55, 0.25 + elapsed * 0.3))
        draw_text_badge(frame, "MUTE", (20, 60), bg_color=(20, 20, 20), fg_color=(200, 200, 200))
        cx, cy = int(w * 0.9), int(h * 0.12)
        cv2.circle(frame, (cx, cy), 28, (30, 30, 30), -1, cv2.LINE_AA)
        cv2.putText(frame, "M", (cx - 12, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2, cv2.LINE_AA)
        cv2.line(frame, (cx - 18, cy + 18), (cx + 18, cy - 18), (80, 80, 255), 3, cv2.LINE_AA)
        return frame

