from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Type

import cv2

from app.config import DEFAULT_CONFIG
from app.detector import Detection, GestureDetector
from app.effects import (
    DislikeEffect,
    GunEffect,
    HeartEffect,
    MiddleFingerEffect,
    MuteEffect,
    OkPulseEffect,
    PeaceEffect,
    RockEffect,
    ThumbsUpEffect,
)
from app.effects.base import BaseEffect
from app.gesture_smoother import GestureSmoother


EFFECT_CLASS_MAP: Dict[str, Type[BaseEffect]] = {
    "like": ThumbsUpEffect,
    "dislike": DislikeEffect,
    "mute": MuteEffect,
    "ok": OkPulseEffect,
    "peace": PeaceEffect,
    "rock": RockEffect,
    "three_gun": GunEffect,
    "middle_finger": MiddleFingerEffect,
    "hand_heart": HeartEffect,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time gesture visual effects app.")
    parser.add_argument("--model", type=str, default=DEFAULT_CONFIG.inference.model_path)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--webcam", type=int, default=DEFAULT_CONFIG.inference.webcam_index)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_CONFIG.inference.input_size)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--window-size", type=int, default=DEFAULT_CONFIG.inference.smoothing_window_size)
    parser.add_argument("--stable-frames", type=int, default=DEFAULT_CONFIG.inference.stable_frames_required)
    parser.add_argument("--cooldown", type=float, default=DEFAULT_CONFIG.inference.trigger_cooldown_sec)
    return parser.parse_args()


def draw_detections(frame, detections: list[Detection]) -> None:
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 120), 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def put_status(
    frame,
    fps: float,
    dominant: Optional[Detection],
    stable_gesture: Optional[str],
    stable_conf: float,
) -> None:
    cv2.putText(frame, f"FPS: {fps:.1f}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 230, 255), 2)
    gesture_txt = "none" if dominant is None else f"{dominant.class_name} ({dominant.confidence:.2f})"
    stable_txt = "none" if stable_gesture is None else f"{stable_gesture} ({stable_conf:.2f})"
    cv2.putText(frame, f"Dominant: {gesture_txt}", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2)
    cv2.putText(frame, f"Stable: {stable_txt}", (12, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2)


def resolve_device(device_arg: str) -> str | int:
    if device_arg.isdigit():
        return int(device_arg)
    return device_arg


def main() -> None:
    args = parse_args()
    model_path = str(Path(args.model))
    detector = GestureDetector(
        model_path=model_path,
        class_id_to_name=DEFAULT_CONFIG.id_to_class,
        per_class_thresholds=DEFAULT_CONFIG.class_thresholds,
        default_threshold=args.conf,
        device=resolve_device(args.device),
    )
    smoother = GestureSmoother(
        per_class_thresholds=DEFAULT_CONFIG.class_thresholds,
        window_size=args.window_size,
        stable_frames_required=args.stable_frames,
        cooldown_sec=args.cooldown,
        ignored_trigger_class=DEFAULT_CONFIG.inference.effect_ignore_class,
    )
    active_effect: Optional[BaseEffect] = None

    cap = cv2.VideoCapture(args.webcam)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open webcam index {args.webcam}")

    prev = time.perf_counter()
    smooth_fps = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.perf_counter()
            dt = max(1e-6, now - prev)
            prev = now
            inst_fps = 1.0 / dt
            if smooth_fps == 0.0:
                smooth_fps = inst_fps
            else:
                alpha = DEFAULT_CONFIG.inference.fps_smoothing_alpha
                smooth_fps = alpha * smooth_fps + (1 - alpha) * inst_fps

            detections = detector.infer(frame, imgsz=args.imgsz)
            dominant = detector.dominant_detection(detections)
            smoothed = smoother.update(dominant, now=now)

            if smoothed.should_trigger and smoothed.stable_gesture:
                effect_cls = EFFECT_CLASS_MAP.get(smoothed.stable_gesture)
                if effect_cls is not None:
                    if active_effect is None or type(active_effect) is not effect_cls:
                        active_effect = effect_cls()
                    active_effect.trigger(dominant, frame.shape)

            draw_detections(frame, detections)
            if active_effect is not None:
                frame = active_effect.update(frame, dt)
                if not active_effect.active:
                    active_effect = None
            put_status(frame, smooth_fps, dominant, smoothed.stable_gesture, smoothed.stable_confidence)

            cv2.imshow("Gesture Effects MVP", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

