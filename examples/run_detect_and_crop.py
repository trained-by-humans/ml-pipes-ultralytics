"""Crop detections, annotate, display, and save one image with YOLO."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from common import EXAMPLE_ASSETS, default_image
from ml_pipes.core import Pipeline
from ml_pipes.standard import Select
from ml_pipes.ultralytics import results, yolo


def build_pipeline(
    model: str,
    output: Path,
    crop_dir: Path,
    *,
    classes: list[int] | None,
    conf: float,
    iou: float,
    imgsz: int,
    device: str | None,
    max_det: int,
    show: bool,
) -> Pipeline[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Build an ObjectCropper-inspired single-image pipeline."""
    return Pipeline(
        [
            yolo.Predict(
                model=model,
                classes=classes,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                max_det=max_det,
            ),
            Select(0),
            results.SaveCrop(crop_dir),
            results.Plot(show=show, save=True, filename=str(output)),
        ],
        auto_validate=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=default_image(),
        help="Input image path. Defaults to .example_assets/bus.jpg.",
    )
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--classes", type=int, nargs="+", default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output", type=Path, default=EXAMPLE_ASSETS / "annotated.jpg")
    parser.add_argument("--crop-dir", type=Path, default=EXAMPLE_ASSETS / "cropped-detections")
    args = parser.parse_args()

    image = cv2.imread(str(args.source))
    if image is None:
        raise RuntimeError(f"Could not load {args.source}")

    pipeline = build_pipeline(
        args.model,
        args.output,
        args.crop_dir,
        classes=args.classes,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        max_det=args.max_det,
        show=args.show,
    )
    pipeline.validate()
    pipeline.describe()
    annotated = pipeline(image)
    print(f"Annotated image shape: {annotated.shape}")
    print(f"Saved {args.output}")
    print(f"Saved crops to {args.crop_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
