from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

from scripts.common import ensure_dir

DEFAULT_CLASSES = [
    "like",
    "dislike",
    "mute",
    "ok",
    "peace",
    "rock",
    "three_gun",
    "middle_finger",
    "hand_heart",
    "no_gesture",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download HaGRID subset with official hagrid download.py")
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--vendor-hagrid", type=Path, default=Path("vendor/hagrid"))
    parser.add_argument("--annotations", action="store_true")
    parser.add_argument("--images", action="store_true")
    parser.add_argument("--no-extract", action="store_true", help="Do not unzip downloaded archives")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def run_hagrid_download(args: argparse.Namespace) -> None:
    download_script = args.vendor_hagrid / "download.py"
    if not download_script.exists():
        raise FileNotFoundError(
            f"Cannot find {download_script}. Clone HaGRID first: "
            "git clone https://github.com/hukenovs/hagrid vendor/hagrid"
        )

    cmd = [sys.executable, str(download_script), "--save_path", str(args.out)]
    if args.annotations:
        cmd.append("--annotations")
    if args.images:
        cmd.append("--dataset")
    if args.images and args.classes:
        cmd.extend(["--targets", *args.classes])

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def extract_zip_archives(out_dir: Path) -> None:
    zips = sorted(out_dir.rglob("*.zip"))
    if not zips:
        print("No zip files found to extract.")
        return

    for zip_path in zips:
        target_dir = out_dir
        with ZipFile(zip_path, "r") as zf:
            print(f"Extracting {zip_path.name} -> {target_dir}")
            zf.extractall(target_dir)


def main() -> None:
    args = parse_args()
    ensure_dir(args.out)

    if not args.annotations and not args.images:
        args.annotations = True
        args.images = True

    if args.skip_existing:
        annotations_dir = args.out / "hagrid_annotations"
        dataset_dir = args.out / "hagrid_dataset"
        if ((not args.annotations) or annotations_dir.exists()) and ((not args.images) or dataset_dir.exists()):
            print("Target data directories already exist. Skipping download.")
            return

    run_hagrid_download(args)
    if not args.no_extract:
        extract_zip_archives(args.out)

    print("Done.")
    print(f"Expected annotations root: {args.out / 'hagrid_annotations'}")
    print(f"Expected images root: {args.out / 'hagrid_dataset'}")


if __name__ == "__main__":
    main()
