"""Blur tracked detections in one image with an ml-pipes Ultralytics pipeline.

Run from the repository root:
    python examples/run_object_blurrer.py
    python examples/run_object_blurrer.py --source path/to/image.jpg --show
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from ultralytics.engine.results import Results
from common import EXAMPLE_ASSETS, default_image
from ml_pipes.core import Pipeline
from ml_pipes.operator import Operator
from ml_pipes.standard import Select
from ml_pipes.ultralytics import yolo

HUMAN_CLASS_ID = 0  # COCO's ``person`` class used by the default YOLO model.


@Operator
class ObjectBlurrer:
    """Blur detected boxes, then render the native result over the blurred image."""

    def __init__(
        self,
        blur_ratio: float = 0.5,
        line_width: int = 2,
        show_boxes: bool = True,
        show_conf: bool = True,
        show_labels: bool = True,
    ) -> None:
        if blur_ratio < 0.1:
            raise ValueError("blur_ratio must be at least 0.1.")
        self.blur_kernel = int(blur_ratio * 100)
        self.line_width = line_width
        self.show_boxes = show_boxes
        self.show_conf = show_conf
        self.show_labels = show_labels

    def __call__(self, results: Results) -> npt.NDArray[np.uint8]:
        image = results.orig_img.copy()
        if results.boxes is not None:
            height, width = image.shape[:2]
            for box in results.boxes.xyxy.cpu().numpy():
                x0, y0, x1, y1 = map(int, box)
                x0, y0 = max(x0, 0), max(y0, 0)
                x1, y1 = min(x1, width), min(y1, height)
                if x0 < x1 and y0 < y1:
                    image[y0:y1, x0:x1] = cv2.blur(
                        image[y0:y1, x0:x1],
                        (self.blur_kernel, self.blur_kernel),
                    )
        return results.plot(
            img=image,
            line_width=self.line_width,
            boxes=self.show_boxes,
            conf=self.show_conf,
            labels=self.show_labels,
            masks=False,
            probs=False,
            color_mode="class",
        )


def build_pipeline(
    model: str,
    *,
    conf: float,
    iou: float,
    imgsz: int,
    device: str | None,
    max_det: int,
    tracker: str,
    blur_ratio: float,
    line_width: int,
    show_boxes: bool,
    show_conf: bool,
    show_labels: bool,
) -> Pipeline[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Build an ObjectBlurrer-inspired tracking and rendering pipeline."""
    return Pipeline(
        [
            yolo.Track(
                model=model,
                persist=True,
                tracker=tracker,
                classes=[HUMAN_CLASS_ID],
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                max_det=max_det,
            ),
            Select(0),
            ObjectBlurrer(
                blur_ratio=blur_ratio,
                line_width=line_width,
                show_boxes=show_boxes,
                show_conf=show_conf,
                show_labels=show_labels,
            ),
        ],
        auto_validate=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", type=Path, default=default_image())
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--blur-ratio", type=float, default=0.5)
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--show-boxes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-conf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output", type=Path, default=EXAMPLE_ASSETS / "blurred.jpg")
    args = parser.parse_args()

    image = cv2.imread(str(args.source))
    if image is None:
        raise RuntimeError(f"Could not load {args.source}")

    pipeline = build_pipeline(
        args.model,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        max_det=args.max_det,
        tracker=args.tracker,
        blur_ratio=args.blur_ratio,
        line_width=args.line_width,
        show_boxes=args.show_boxes,
        show_conf=args.show_conf,
        show_labels=args.show_labels,
    )
    blurred = pipeline(image)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), blurred):
        raise RuntimeError(f"Could not save {args.output}")
    if args.show:
        cv2.imshow("Object Blurrer", blurred)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
