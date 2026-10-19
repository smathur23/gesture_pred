from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Type

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

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
from app.model_resolver import resolve_model_path


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
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_CONFIG.inference.model_path,
        help="Path to model weights (.pt). Use 'auto' to pick latest runs/detect/*/weights/best.pt.",
    )
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--webcam", type=int, default=DEFAULT_CONFIG.inference.webcam_index)
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video source: webcam index (e.g. 0) or file path. Overrides --webcam if set.",
    )
    parser.add_argument("--imgsz", type=int, default=DEFAULT_CONFIG.inference.input_size)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--window-size", type=int, default=DEFAULT_CONFIG.inference.smoothing_window_size)
    parser.add_argument("--stable-frames", type=int, default=DEFAULT_CONFIG.inference.stable_frames_required)
    parser.add_argument("--cooldown", type=float, default=DEFAULT_CONFIG.inference.trigger_cooldown_sec)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--mirror", action="store_true", help="Mirror preview horizontally (selfie view).")
    parser.add_argument("--record-out", type=str, default=None, help="Optional output video path (mp4).")
    parser.add_argument("--show-controls", action="store_true", help="Draw keyboard controls on frame.")
    parser.add_argument("--list-cameras", action="store_true", help="Probe and print available camera indices, then exit.")
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


def parse_source(source_arg: Optional[str], webcam_index: int) -> str | int:
    if source_arg is None:
        return webcam_index
    return int(source_arg) if source_arg.isdigit() else source_arg


def open_capture(source: str | int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def probe_cameras(max_index: int = 10, width: int = 640, height: int = 480) -> List[int]:
    # Fast path for Linux/WSL: if no /dev/video* exists, probing indices only produces noisy backend warnings.
    if Path("/dev").exists() and not list(Path("/dev").glob("video*")):
        return []

    found: List[int] = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if cap.isOpened():
            found.append(idx)
        cap.release()
    return found


def is_wsl() -> bool:
    if "WSL_DISTRO_NAME" in os.environ:
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False


def format_capture_error(source: str | int) -> str:
    lines = [f"Unable to open source: {source}"]
    if isinstance(source, int):
        found = probe_cameras(max_index=8)
        lines.append(f"Detected camera indices: {found if found else 'none'}")
    if is_wsl():
        lines.append("Detected WSL2 environment and no webcam is exposed to Linux runtime.")
        lines.append("Run from native Windows Python/Conda for webcam access, or attach a USB webcam to WSL via usbipd.")
        lines.append("You can still test app logic with a video file: --source path/to/video.mp4")
    return "\n".join(lines)


def open_writer(record_path: str, frame: np.ndarray, fallback_fps: float = 30.0) -> cv2.VideoWriter:
    out_path = Path(record_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(out_path), fourcc, fallback_fps, (w, h))


def draw_controls(frame: np.ndarray) -> None:
    lines = [
        "q: quit",
        "b: boxes on/off",
        "e: effects on/off",
        "m: mirror on/off",
        "h: controls on/off",
    ]
    y = frame.shape[0] - 14
    for txt in reversed(lines):
        cv2.putText(frame, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (225, 225, 225), 1, cv2.LINE_AA)
        y -= 18


def main() -> None:
    args = parse_args()
    if args.list_cameras:
        found = probe_cameras(max_index=12)
        print(f"[app] available camera indices: {found if found else 'none'}")
        if is_wsl() and not found:
            print("[app] WSL2 detected and no cameras are visible in Linux runtime.")
        return

    model_path = resolve_model_path(
        requested_model=args.model,
        project_root=DEFAULT_CONFIG.paths.project_root,
        runs_root=DEFAULT_CONFIG.paths.runs_root,
    )
    print(f"[app] model: {model_path}")

    source = parse_source(args.source, args.webcam)
    print(f"[app] source: {source}")
    print("[app] controls: q quit | b boxes | e effects | m mirror | h help")

    detector = GestureDetector(
        model_path=str(model_path),
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

    cap = open_capture(source=source, width=args.width, height=args.height)
    if not cap.isOpened():
        raise RuntimeError(format_capture_error(source))

    writer: Optional[cv2.VideoWriter] = None
    show_boxes = True
    enable_effects = True
    show_help = args.show_controls
    mirror = args.mirror

    prev = time.perf_counter()
    smooth_fps = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)

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

            if enable_effects and smoothed.should_trigger and smoothed.stable_gesture:
                effect_cls = EFFECT_CLASS_MAP.get(smoothed.stable_gesture)
                if effect_cls is not None:
                    if active_effect is None or type(active_effect) is not effect_cls:
                        active_effect = effect_cls()
                    active_effect.trigger(dominant, frame.shape)

            if show_boxes:
                draw_detections(frame, detections)
            if enable_effects and active_effect is not None:
                frame = active_effect.update(frame, dt)
                if not active_effect.active:
                    active_effect = None
            elif not enable_effects:
                active_effect = None
            put_status(frame, smooth_fps, dominant, smoothed.stable_gesture, smoothed.stable_confidence)
            if show_help:
                draw_controls(frame)

            if args.record_out:
                if writer is None:
                    writer = open_writer(args.record_out, frame, fallback_fps=max(1.0, min(120.0, smooth_fps or 30.0)))
                    print(f"[app] recording to: {args.record_out}")
                writer.write(frame)

            cv2.imshow("Gesture Effects MVP", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("b"):
                show_boxes = not show_boxes
            if key == ord("e"):
                enable_effects = not enable_effects
            if key == ord("m"):
                mirror = not mirror
            if key == ord("h"):
                show_help = not show_help
    finally:
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
