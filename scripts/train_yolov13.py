from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv13-S on prepared HaGRID subset.")
    parser.add_argument("--data", type=Path, default=Path("data/hagrid_gestures.yaml"))
    parser.add_argument("--model", type=str, default="yolov13s.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name", type=str, default="hagrid_subset")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def resolve_model(model_arg: str) -> tuple[str, str | None]:
    model_path = Path(model_arg)
    if model_path.exists():
        return str(model_path), None
    vendor_small_yaml = Path("vendor/yolov13/ultralytics/cfg/models/v13/yolov13s.yaml")
    vendor_yaml = Path("vendor/yolov13/ultralytics/cfg/models/v13/yolov13.yaml")
    if model_arg.lower() in {"yolov13s", "yolov13s.pt", "yolov13.yaml"}:
        if vendor_small_yaml.exists():
            return str(vendor_small_yaml), None
        if vendor_yaml.exists():
            return str(vendor_yaml), "s"
    return model_arg, None


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "YOLO package unavailable. Install vendor/yolov13 first: pip install -e vendor/yolov13"
        ) from exc

    model_source, scale_hint = resolve_model(args.model)
    model = YOLO(model_source)
    train_kwargs = dict(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        verbose=True,
    )
    if scale_hint is not None:
        train_kwargs["scale"] = scale_hint
    train_results = model.train(**train_kwargs)
    print("Train results:", train_results)

    val_results = model.val(data=str(args.data), imgsz=args.imgsz, device=args.device)
    print("Val results:", val_results)


if __name__ == "__main__":
    main()
