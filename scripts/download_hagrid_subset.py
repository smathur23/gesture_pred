from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile, is_zipfile

try:
    from scripts.common import ensure_dir
except ModuleNotFoundError:
    from common import ensure_dir

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
    parser.add_argument("--force-python-downloader", action="store_true", help="Bypass vendor downloader and use resumable Python downloader")
    parser.add_argument("--local-archives-dir", type=Path, help="Optional dir containing ann.zip and/or class zip files")
    parser.add_argument("--local-annotations-dir", type=Path, help="Optional local extracted hagrid_annotations directory")
    parser.add_argument("--local-images-dir", type=Path, help="Optional local extracted hagrid_dataset directory")
    parser.add_argument("--ingest-mode", choices=["auto", "symlink", "copy"], default="auto")
    return parser.parse_args()


def run_hagrid_download(args: argparse.Namespace) -> None:
    download_script = args.vendor_hagrid / "download.py"
    if not download_script.exists():
        raise FileNotFoundError(
            f"Cannot find {download_script}. Clone HaGRID first: "
            "git clone https://github.com/hukenovs/hagrid vendor/hagrid"
        )

    if has_wget() and not args.force_python_downloader and not has_partial_archives(args):
        cmd = [sys.executable, str(download_script), "--save_path", str(args.out)]
        if args.annotations:
            cmd.append("--annotations")
        if args.images:
            cmd.append("--dataset")
        if args.images and args.classes:
            cmd.extend(["--targets", *args.classes])
        print("Running:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError as exc:
            print(f"Vendor downloader failed ({exc}). Falling back to Python URL downloader.")
    else:
        if args.force_python_downloader:
            print("Using Python downloader because --force-python-downloader was set.")
        elif has_partial_archives(args):
            print("Using Python downloader because partial zip files were detected (resumable mode).")
        else:
            print("System wget not found. Falling back to Python URL downloader.")

    urls = load_hagrid_urls(download_script)
    if args.images:
        for class_name in args.classes:
            download_url = urls.get(class_name)
            if not download_url:
                print(f"Skipping unknown class {class_name}")
                continue
            dst = args.out / f"{class_name}.zip"
            if args.skip_existing and dst.exists():
                print(f"Skipping existing: {dst.name}")
                continue
            print(f"Downloading {class_name}.zip")
            try:
                download_file(download_url, dst)
            except (HTTPError, URLError, OSError) as exc:
                raise RuntimeError(
                    f"Failed to download {class_name}.zip from {download_url}: {exc}. "
                    "Use --local-archives-dir or --local-images-dir to ingest local copies."
                ) from exc
    if args.annotations:
        ann_url = urls["annotations"]
        ann_dst = args.out / "ann.zip"
        if not (args.skip_existing and ann_dst.exists()):
            print("Downloading ann.zip")
            try:
                download_file(ann_url, ann_dst)
            except (HTTPError, URLError, OSError) as exc:
                raise RuntimeError(
                    f"Failed to download ann.zip from {ann_url}: {exc}. "
                    "Use --local-archives-dir or --local-annotations-dir to ingest local copies."
                ) from exc


def has_partial_archives(args: argparse.Namespace) -> bool:
    candidates: list[Path] = []
    if args.annotations:
        candidates.append(args.out / "ann.zip")
    if args.images:
        candidates.extend((args.out / f"{class_name}.zip") for class_name in args.classes)
    for path in candidates:
        if path.exists() and path.is_file() and path.stat().st_size > 0:
            return True
    return False


def ingest_local_archives(args: argparse.Namespace) -> None:
    if not args.local_archives_dir:
        return
    src_dir = args.local_archives_dir.resolve()
    if not src_dir.exists():
        raise FileNotFoundError(f"Local archives dir does not exist: {src_dir}")
    print(f"Ingesting local archives from {src_dir}")

    if args.annotations:
        src_ann = src_dir / "ann.zip"
        if src_ann.exists():
            dst_ann = args.out / "ann.zip"
            if not (args.skip_existing and dst_ann.exists()):
                shutil.copy2(src_ann, dst_ann)
                print(f"Copied {src_ann.name} -> {dst_ann}")

    if args.images:
        for class_name in args.classes:
            src_zip = src_dir / f"{class_name}.zip"
            if not src_zip.exists():
                continue
            dst_zip = args.out / src_zip.name
            if args.skip_existing and dst_zip.exists():
                continue
            shutil.copy2(src_zip, dst_zip)
            print(f"Copied {src_zip.name} -> {dst_zip}")


def place_local_tree(src: Path, dst: Path, mode: str) -> None:
    src_resolved = src.resolve()
    if not src_resolved.exists():
        raise FileNotFoundError(f"Local source path does not exist: {src_resolved}")
    if dst.exists():
        print(f"Local target already exists, skipping: {dst}")
        return

    try_modes = ["symlink", "copy"] if mode == "auto" else [mode]
    last_error: Exception | None = None
    for attempt in try_modes:
        try:
            if attempt == "symlink":
                os.symlink(src_resolved, dst, target_is_directory=True)
            else:
                shutil.copytree(src_resolved, dst)
            print(f"Ingested local directory via {attempt}: {src_resolved} -> {dst}")
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"Failed to ingest local directory {src_resolved} -> {dst}: {last_error}")


def ingest_local_directories(args: argparse.Namespace) -> None:
    if args.local_annotations_dir and args.annotations:
        place_local_tree(args.local_annotations_dir, args.out / "hagrid_annotations", args.ingest_mode)
    if args.local_images_dir and args.images:
        place_local_tree(args.local_images_dir, args.out / "hagrid_dataset", args.ingest_mode)


def requested_archives(args: argparse.Namespace) -> list[Path]:
    archives: list[Path] = []
    if args.annotations:
        archives.append(args.out / "ann.zip")
    if args.images:
        archives.extend((args.out / f"{class_name}.zip") for class_name in args.classes)
    return archives


def extract_zip_archives(archives: list[Path], out_dir: Path) -> None:
    zips = [p for p in archives if p.exists()]
    if not zips:
        print("No zip files found to extract.")
        return

    for zip_path in zips:
        if not is_zipfile(zip_path):
            print(f"Skipping non-zip or partial archive: {zip_path}")
            continue
        target_dir = out_dir
        try:
            with ZipFile(zip_path, "r") as zf:
                print(f"Extracting {zip_path.name} -> {target_dir}")
                zf.extractall(target_dir)
        except BadZipFile:
            print(f"Skipping invalid archive (corrupt/partial): {zip_path}")


def normalize_extracted_layout(args: argparse.Namespace) -> None:
    annotations_dir = args.out / "annotations"
    hagrid_annotations_dir = args.out / "hagrid_annotations"
    if args.annotations and annotations_dir.exists() and not hagrid_annotations_dir.exists():
        annotations_dir.rename(hagrid_annotations_dir)
        print(f"Normalized annotations path: {annotations_dir} -> {hagrid_annotations_dir}")

    if not args.images:
        return

    dataset_dir = args.out / "hagrid_dataset"
    ensure_dir(dataset_dir)
    for class_name in args.classes:
        legacy_class_dir = args.out / class_name
        class_dir = dataset_dir / class_name
        if legacy_class_dir.exists() and legacy_class_dir.is_dir() and not class_dir.exists():
            shutil.move(str(legacy_class_dir), str(class_dir))
            print(f"Normalized class path: {legacy_class_dir} -> {class_dir}")


def has_wget() -> bool:
    return shutil.which("wget") is not None


def load_hagrid_urls(download_script: Path) -> dict:
    spec = spec_from_file_location("hagrid_download_mod", str(download_script))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {download_script}")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {key: normalize_url(str(value)) for key, value in dict(mod.urls).items()}


def normalize_url(url: str) -> str:
    if "://" not in url:
        return url
    scheme, remainder = url.split("://", maxsplit=1)
    while "//" in remainder:
        remainder = remainder.replace("//", "/")
    return f"{scheme}://{remainder}"


def download_file(url: str, dst: Path, chunk_size: int = 8 * 1024 * 1024) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    downloaded = dst.stat().st_size if dst.exists() else 0
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "*/*",
        "Connection": "keep-alive",
    }
    if downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"
        print(f"Resuming {dst.name} from {downloaded} bytes")

    req = Request(url, headers=headers)
    with urlopen(req) as resp:
        status = getattr(resp, "status", None)
        if downloaded > 0 and status == 200:
            print(f"Server ignored range for {dst.name}; restarting file.")
            dst.unlink(missing_ok=True)
            downloaded = 0
            req = Request(url, headers={k: v for k, v in headers.items() if k != "Range"})
            resp.close()
            with urlopen(req) as resp2, dst.open("wb") as out:
                while True:
                    chunk = resp2.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
            return

        mode = "ab" if downloaded > 0 else "wb"
        with dst.open(mode) as out:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)


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

    ingest_local_directories(args)
    ingest_local_archives(args)

    already_have_annotations = (args.out / "hagrid_annotations").exists()
    already_have_images = (args.out / "hagrid_dataset").exists()

    should_download = True
    if args.skip_existing and ((not args.annotations) or already_have_annotations) and ((not args.images) or already_have_images):
        should_download = False

    if should_download:
        run_hagrid_download(args)
    else:
        print("Requested dataset directories already exist after local ingest. Skipping remote download.")

    if not args.no_extract:
        extract_zip_archives(requested_archives(args), args.out)
    normalize_extracted_layout(args)

    print("Done.")
    print(f"Expected annotations root: {args.out / 'hagrid_annotations'}")
    print(f"Expected images root: {args.out / 'hagrid_dataset'}")


if __name__ == "__main__":
    main()
