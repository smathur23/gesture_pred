from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

TWO_HANDED_CLASSES = {
    "hand_heart",
    "hand_heart2",
    "thumb_index2",
    "timeout",
    "holy",
    "take_picture",
    "xsign",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clip01(val: float) -> float:
    return max(0.0, min(1.0, float(val)))


def coco_xywh_to_yolo_xywh(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = [clip01(v) for v in bbox]
    cx = clip01(x + (w * 0.5))
    cy = clip01(y + (h * 0.5))
    return cx, cy, clip01(w), clip01(h)


def iter_annotation_items(annotation_path: Path) -> Iterator[Tuple[str, dict]]:
    try:
        import ijson  # type: ignore
    except ImportError:
        ijson = None

    if ijson is not None:
        with annotation_path.open("rb") as fh:
            for image_id, row in ijson.kvitems(fh, ""):
                if isinstance(row, dict):
                    yield str(image_id), row
        return

    with annotation_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        for image_id, row in data.items():
            if isinstance(row, dict):
                yield str(image_id), row
    elif isinstance(data, list):
        for idx, row in enumerate(data):
            if isinstance(row, dict):
                image_id = str(row.get("image_id", idx))
                yield image_id, row


def pick_bboxes_for_class(row: dict, target_class: str) -> List[Tuple[str, Tuple[float, float, float, float]]]:
    boxes: List[Tuple[str, Tuple[float, float, float, float]]] = []
    if target_class in TWO_HANDED_CLASSES:
        united_boxes = row.get("united_bbox") or []
        united_labels = row.get("united_label") or []
        if isinstance(united_boxes, list) and isinstance(united_labels, list):
            for bbox, label in zip(united_boxes, united_labels):
                if label == target_class and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    boxes.append((target_class, coco_xywh_to_yolo_xywh(bbox)))
            if boxes:
                return boxes

    labels = row.get("labels") or []
    bboxes = row.get("bboxes") or []
    if not isinstance(labels, list) or not isinstance(bboxes, list):
        return boxes

    for label, bbox in zip(labels, bboxes):
        if label != target_class:
            continue
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        boxes.append((target_class, coco_xywh_to_yolo_xywh(bbox)))
    return boxes


def find_image_path(images_root: Path, class_name: str, image_id: str) -> Optional[Path]:
    class_dir = images_root / class_name
    for ext in IMAGE_EXTENSIONS:
        candidate = class_dir / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    if class_dir.exists():
        matches = list(class_dir.glob(f"{image_id}.*"))
        if matches:
            return matches[0]
    return None


def safe_filename(class_name: str, image_id: str, suffix: str) -> str:
    cleaned = image_id.replace("/", "_").replace("\\", "_")
    return f"{class_name}__{cleaned}{suffix}"


def link_or_copy(src: Path, dst: Path, mode: str = "auto") -> str:
    ensure_dir(dst.parent)
    if dst.exists():
        return "exists"

    attempts: List[str]
    if mode == "auto":
        attempts = ["symlink", "hardlink", "copy"]
    elif mode in {"symlink", "hardlink", "copy"}:
        attempts = [mode]
    else:
        raise ValueError(f"Unsupported link mode: {mode}")

    last_error: Optional[Exception] = None
    for attempt in attempts:
        try:
            if attempt == "symlink":
                os.symlink(src, dst)
                return attempt
            if attempt == "hardlink":
                os.link(src, dst)
                return attempt
            shutil.copy2(src, dst)
            return attempt
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
    raise RuntimeError(f"Unable to place image {src} -> {dst}: {last_error}")


def load_manifest_rows(manifest_path: Path) -> Iterator[dict]:
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    ensure_dir(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def parse_csv_int_map(items: Sequence[str]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for item in items:
        key, value = item.split("=", maxsplit=1)
        result[key.strip()] = int(value.strip())
    return result

