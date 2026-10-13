"""Process a photo archive one image at a time or in YOLO batches.

Run from the repository root:
    python examples/run_batch_predict.py
    python examples/run_batch_predict.py --source path/to/images --mode concurrent
"""
from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import numpy.typing as npt
from common import EXAMPLE_ASSETS
from ml_pipes.core import Pipeline
from ml_pipes.standard import Gather, Map, Scatter, Select
from ml_pipes.ultralytics import yolo
from ml_pipes.vision import Decode, ImagePayload, LoadFile


def default_archive() -> Path:
    """Download the tutorial's 100 COCO128 JPEGs once, then reuse them."""
    destination = EXAMPLE_ASSETS / "coco-sample"
    if any(destination.glob("*.jpg")):
        return destination

    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
    print(f"Downloading sample images to {destination}")
    with urlopen(url, timeout=60) as response:
        archive_bytes = response.read()

    destination.mkdir(parents=True, exist_ok=True)
    with ZipFile(BytesIO(archive_bytes)) as archive:
        images = sorted(
            name
            for name in archive.namelist()
            if "/images/train2017/" in name and name.lower().endswith(".jpg")
        )[:100]
        for name in images:
            output = destination / Path(name).name
            output.write_bytes(archive.read(name))
    return destination


def image_array(payload: ImagePayload) -> npt.NDArray[np.uint8]:
    """Pass an original-size BGR image to YOLO for native preprocessing."""
    return payload.array


def build_pipeline(
    model: str,
    *,
    mode: str,
    workers: int,
    conf: float,
    imgsz: int,
    device: str | None,
) -> Pipeline:
    """Build the selected tutorial stage, reusing one model across calls."""
    if mode == "single":
        return Pipeline(
            [
                yolo.Predict(model=model, conf=conf, imgsz=imgsz, device=device),
                Select(0),
            ],
            auto_validate=True,
        )
    if mode == "batch":
        return Pipeline(
            [yolo.Predict(model=model, conf=conf, imgsz=imgsz, device=device)],
            auto_validate=True,
        )
    if mode == "concurrent":
        return Pipeline(
            [
                Scatter(max_concurrency=workers),
                LoadFile(),
                Decode(),
                Map(image_array),
                Gather(),
                yolo.Predict(model=model, conf=conf, imgsz=imgsz, device=device),
            ],
            auto_validate=True,
        )
    raise ValueError(f"Unknown mode: {mode}")


def positive_int(value: str) -> int:
    """Require a positive batch size or worker count at the CLI boundary."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Image directory. Defaults to 100 cached COCO128 sample images.",
    )
    parser.add_argument("--pattern", default="*.jpg", help="Image glob within --source.")
    parser.add_argument(
        "--mode",
        choices=("single", "batch", "concurrent"),
        default="concurrent",
        help="Tutorial stage to run (default: concurrent).",
    )
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--workers", type=positive_int, default=4, help="Concurrent decode workers.")
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    image_dir = args.source if args.source is not None else default_archive()
    sources = sorted(path for path in image_dir.glob(args.pattern) if path.is_file())
    if not sources:
        parser.error(f"No images matching {args.pattern!r} in {image_dir}")

    pipeline = build_pipeline(
        args.model,
        mode=args.mode,
        workers=args.workers,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )
    pipeline.validate()
    pipeline.describe()

    print(f"Processing {len(sources)} images with {args.mode} mode")
    if args.mode == "single":
        for source in sources:
            result = pipeline(str(source))
            print(f"{source.name}: {result.verbose().strip()}")
    else:
        for start in range(0, len(sources), args.batch_size):
            paths = sources[start : start + args.batch_size]
            predictions = pipeline([str(path) for path in paths])
            for path, result in zip(paths, predictions):
                print(f"{path.name}: {result.verbose().strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
