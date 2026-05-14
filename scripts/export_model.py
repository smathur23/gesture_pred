from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained YOLO model to deployment formats.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "engine", "torchscript"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--half", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"Missing weights: {args.weights}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "YOLO package unavailable. Install vendor/yolov13 first: pip install -e vendor/yolov13"
        ) from exc

    model = YOLO(str(args.weights))
    exported_path = model.export(format=args.format, imgsz=args.imgsz, device=args.device, half=args.half)
    print(f"Exported model: {exported_path}")


if __name__ == "__main__":
    main()

