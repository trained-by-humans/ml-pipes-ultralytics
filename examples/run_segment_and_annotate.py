"""Segment, annotate, display, and save an image with a YOLO segmentation model.

Run from the repository root:
    python examples/run_segment_and_annotate.py
    python examples/run_segment_and_annotate.py --source path/to/image.jpg --show
"""
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
    *,
    classes: list[int] | None,
    conf: float,
    iou: float,
    imgsz: int,
    device: str | None,
    max_det: int,
    line_width: int,
    show_boxes: bool,
    show_conf: bool,
    show_labels: bool,
    show: bool,
) -> Pipeline[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Build an InstanceSegmentation-inspired single-image pipeline."""
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
            results.Plot(
                line_width=line_width,
                boxes=show_boxes,
                conf=show_conf,
                labels=show_labels,
                show=show,
                save=True,
                filename=str(output),
                color_mode="instance",
            ),
        ],
        auto_validate=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=default_image(),
        help="Input image path. Defaults to .example_assets/bus.jpg.",
    )
    parser.add_argument("--model", default="yolo26n-seg.pt")
    parser.add_argument("--classes", type=int, nargs="+", default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--show-boxes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-conf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output", type=Path, default=EXAMPLE_ASSETS / "segmented.jpg")
    args = parser.parse_args()

    image = cv2.imread(str(args.source))
    if image is None:
        raise RuntimeError(f"Could not load {args.source}")

    pipeline = build_pipeline(
        args.model,
        args.output,
        classes=args.classes,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        max_det=args.max_det,
        line_width=args.line_width,
        show_boxes=args.show_boxes,
        show_conf=args.show_conf,
        show_labels=args.show_labels,
        show=args.show,
    )
    pipeline.validate()
    pipeline.describe()
    annotated = pipeline(image)
    print(f"Annotated image shape: {annotated.shape}")
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
