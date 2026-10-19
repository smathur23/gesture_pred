from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.config import DEFAULT_CONFIG
from app.detector import GestureDetector
from app.gesture_smoother import GestureSmoother
from app.model_resolver import resolve_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark webcam gesture inference and stability.")
    parser.add_argument("--model", type=str, default=DEFAULT_CONFIG.inference.model_path)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--webcam", type=int, default=0)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--stable-frames", type=int, default=6)
    parser.add_argument("--cooldown", type=float, default=1.0)
    return parser.parse_args()


def resolve_device(device_arg: str) -> str | int:
    return int(device_arg) if device_arg.isdigit() else device_arg


def parse_source(source_arg: str | None, webcam_index: int) -> str | int:
    if source_arg is None:
        return webcam_index
    return int(source_arg) if source_arg.isdigit() else source_arg


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(
        requested_model=args.model,
        project_root=DEFAULT_CONFIG.paths.project_root,
        runs_root=DEFAULT_CONFIG.paths.runs_root,
    )
    source = parse_source(args.source, args.webcam)
    print(f"[benchmark] model: {model_path}")
    print(f"[benchmark] source: {source}")

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
        ignored_trigger_class="no_gesture",
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {source}")

    t0 = time.perf_counter()
    frame_count = 0
    trigger_count = 0
    dominant_counter: Counter[str] = Counter()
    stable_counter: Counter[str] = Counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1
        detections = detector.infer(frame, imgsz=args.imgsz)
        dominant = detector.dominant_detection(detections)
        if dominant is not None:
            dominant_counter[dominant.class_name] += 1

        smoothed = smoother.update(dominant)
        if smoothed.stable_gesture:
            stable_counter[smoothed.stable_gesture] += 1
        if smoothed.should_trigger:
            trigger_count += 1

        if (time.perf_counter() - t0) >= args.seconds:
            break

    cap.release()
    elapsed = max(1e-6, time.perf_counter() - t0)
    fps = frame_count / elapsed
    false_trigger_rate = 0.0
    if frame_count > 0:
        false_trigger_rate = trigger_count / frame_count

    print(f"Elapsed seconds: {elapsed:.2f}")
    print(f"Frames: {frame_count}")
    print(f"FPS: {fps:.2f}")
    print(f"Triggers: {trigger_count}")
    print(f"Trigger/frame ratio: {false_trigger_rate:.4f}")
    print("Dominant gesture counts:", dict(dominant_counter))
    print("Stable gesture counts:", dict(stable_counter))


if __name__ == "__main__":
    main()
