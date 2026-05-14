from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from time import monotonic
from typing import Deque, Dict, Optional

from app.detector import Detection


@dataclass
class SmoothedState:
    stable_gesture: Optional[str]
    stable_confidence: float
    should_trigger: bool


class GestureSmoother:
    def __init__(
        self,
        per_class_thresholds: Dict[str, float],
        window_size: int = 10,
        stable_frames_required: int = 6,
        cooldown_sec: float = 1.0,
        ignored_trigger_class: str = "no_gesture",
    ) -> None:
        self.per_class_thresholds = per_class_thresholds
        self.window: Deque[tuple[str, float]] = deque(maxlen=window_size)
        self.stable_frames_required = stable_frames_required
        self.cooldown_sec = cooldown_sec
        self.ignored_trigger_class = ignored_trigger_class
        self.last_trigger_time: Dict[str, float] = {}

    def update(self, detection: Optional[Detection], now: Optional[float] = None) -> SmoothedState:
        ts = monotonic() if now is None else now
        if detection is None:
            self.window.append(("no_gesture", 1.0))
        else:
            self.window.append((detection.class_name, detection.confidence))
        if not self.window:
            return SmoothedState(stable_gesture=None, stable_confidence=0.0, should_trigger=False)

        gestures = [gesture for gesture, _ in self.window]
        counts = Counter(gestures)
        stable_gesture, stable_count = counts.most_common(1)[0]
        stable_confidence = self._mean_confidence_for(stable_gesture)

        threshold = self.per_class_thresholds.get(stable_gesture, 0.4)
        is_stable = stable_count >= self.stable_frames_required and stable_confidence >= threshold
        if not is_stable:
            return SmoothedState(stable_gesture=None, stable_confidence=0.0, should_trigger=False)

        if stable_gesture == self.ignored_trigger_class:
            return SmoothedState(
                stable_gesture=stable_gesture,
                stable_confidence=stable_confidence,
                should_trigger=False,
            )

        if stable_gesture not in self.last_trigger_time:
            should_trigger = True
        else:
            last_time = self.last_trigger_time[stable_gesture]
            should_trigger = (ts - last_time) >= self.cooldown_sec
        if should_trigger:
            self.last_trigger_time[stable_gesture] = ts
        return SmoothedState(
            stable_gesture=stable_gesture,
            stable_confidence=stable_confidence,
            should_trigger=should_trigger,
        )

    def _mean_confidence_for(self, gesture: str) -> float:
        vals = [conf for cls, conf in self.window if cls == gesture]
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))
