from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize random YOLO labels to verify conversion.")
    parser.add_argument("--data", type=Path, default=Path("data/hagrid_gestures.yaml"))
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("runs/label_viz"))
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_dataset_yaml(path: Path) -> tuple[Path, Dict[int, str], str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = (path.parent / data["path"]).resolve()
    names_raw = data["names"]
    names = {int(k): str(v) for k, v in names_raw.items()} if isinstance(names_raw, dict) else dict(enumerate(names_raw))
    return root, names, data["train"]


def read_label_file(path: Path) -> List[Tuple[int, float, float, float, float]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, cx, cy, w, h = parts
        rows.append((int(cls), float(cx), float(cy), float(w), float(h)))
    return rows


def draw_yolo_boxes(image, labels: List[Tuple[int, float, float, float, float]], names: Dict[int, str]) -> None:
    h, w = image.shape[:2]
    for cls_id, cx, cy, bw, bh in labels:
        x1 = int((cx - bw / 2.0) * w)
        y1 = int((cy - bh / 2.0) * h)
        x2 = int((cx + bw / 2.0) * w)
        y2 = int((cy + bh / 2.0) * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 120), 2)
        label = names.get(cls_id, str(cls_id))
        cv2.putText(image, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def make_contact_sheet(images: List, grid_cols: int = 4, cell_size: tuple[int, int] = (480, 270)):
    if not images:
        return None
    cols = grid_cols
    rows = (len(images) + cols - 1) // cols
    cw, ch = cell_size
    sheet = np.zeros((rows * ch, cols * cw, 3), dtype=np.uint8)
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        resized = cv2.resize(img, (cw, ch))
        sheet[r * ch : (r + 1) * ch, c * cw : (c + 1) * cw] = resized
    return sheet


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    root, names, _ = load_dataset_yaml(args.data)
    image_dir = root / "images" / args.split
    label_dir = root / "labels" / args.split

    image_files = list(image_dir.glob("*.*"))
    if not image_files:
        raise FileNotFoundError(f"No images found under {image_dir}")

    selected = random.sample(image_files, k=min(args.samples, len(image_files)))
    vis_images = []
    for image_path in selected:
        label_path = label_dir / f"{image_path.stem}.txt"
        labels = read_label_file(label_path)
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        draw_yolo_boxes(img, labels, names)
        cv2.putText(img, image_path.name, (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        vis_images.append(img)
        if args.show:
            cv2.imshow("label_viz", img)
            if cv2.waitKey(120) & 0xFF == ord("q"):
                break

    args.out.mkdir(parents=True, exist_ok=True)
    sheet = make_contact_sheet(vis_images)
    if sheet is not None:
        out_path = args.out / f"{args.split}_samples.jpg"
        cv2.imwrite(str(out_path), sheet)
        print(f"Saved contact sheet: {out_path}")

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
