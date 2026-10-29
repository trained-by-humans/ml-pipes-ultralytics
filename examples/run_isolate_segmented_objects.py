"""Export one PNG per segmented object, with transparency or a black background.

Run from the repository root:
    python examples/run_isolate_segmented_objects.py
    python examples/run_isolate_segmented_objects.py --crop --background black
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import numpy.typing as npt
from ultralytics.engine.results import Results

from common import EXAMPLE_ASSETS, default_image
from ml_pipes.core import Pipeline
from ml_pipes.operator import Operator
from ml_pipes.standard import Select
from ml_pipes.ultralytics import yolo


@Operator
class IsolateObjects:
    """Rasterize native mask polygons and save isolated objects as PNG files."""

    def __init__(
        self,
        output_dir: Path,
        *,
        source_name: str = "image",
        crop: bool = False,
        background: str = "transparent",
    ) -> None:
        if background not in {"transparent", "black"}:
            raise ValueError("background must be 'transparent' or 'black'.")
        self.output_dir = Path(output_dir)
        self.source_name = self._filename_part(source_name)
        self.crop = crop
        self.background = background

    @staticmethod
    def _filename_part(value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_-]+", "_", value)[:80].strip("_") or "unknown"

    def __call__(self, result: Results) -> list[Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            return []

        image = result.orig_img
        height, width = image.shape[:2]
        native_boxes = result.boxes.cpu().numpy()
        boxes = native_boxes.xyxy
        classes = native_boxes.cls
        # masks.xy removes inference letterboxing and scales to orig_img pixels.
        polygons = result.masks.xy
        run_id = uuid4().hex
        paths = []
        for index, (polygon, box, class_id) in enumerate(zip(polygons, boxes, classes)):
            if len(polygon) < 3:
                continue
            mask = np.zeros((height, width), dtype=np.uint8)
            contour = np.rint(polygon).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [contour], 255)
            if not mask.any():
                continue

            isolated = cv2.bitwise_and(image, image, mask=mask)
            if self.background == "transparent":
                isolated = cv2.cvtColor(isolated, cv2.COLOR_BGR2BGRA)
                isolated[:, :, 3] = mask
            if self.crop:
                x0, y0 = np.clip(np.floor(box[:2]), [0, 0], [width, height]).astype(int)
                x1, y1 = np.clip(np.ceil(box[2:]), [0, 0], [width, height]).astype(int)
                if x0 >= x1 or y0 >= y1:
                    continue
                isolated = isolated[y0:y1, x0:x1]

            class_id = int(class_id)
            label = self._filename_part(result.names[class_id])
            path = self.output_dir / (
                f"{self.source_name}_{run_id}_object-{index:03d}_class-{class_id}-{label}.png"
            )
            if not cv2.imwrite(str(path), isolated):
                raise OSError(f"Could not save {path}")
            paths.append(path)
        return paths


def build_pipeline(
    model: str = "yolo26n-seg.pt",
    output_dir: Path = EXAMPLE_ASSETS / "isolated-objects",
    *,
    source_name: str = "image",
    classes: list[int] | None = None,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
    device: str | None = None,
    max_det: int = 300,
    crop: bool = False,
    background: str = "transparent",
) -> Pipeline[npt.NDArray[np.uint8], list[Path]]:
    """Predict instance masks, select the image result, and export its objects."""
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
            IsolateObjects(output_dir, source_name=source_name, crop=crop, background=background),
        ],
        auto_validate=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", type=Path, default=default_image())
    parser.add_argument("--output-dir", type=Path, default=EXAMPLE_ASSETS / "isolated-objects")
    parser.add_argument("--model", default="yolo26n-seg.pt")
    parser.add_argument("--classes", type=int, nargs="+", default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--crop", action="store_true", help="Crop each PNG to its bounded detection box.")
    parser.add_argument("--background", choices=("transparent", "black"), default="transparent")
    args = parser.parse_args()

    image = cv2.imread(str(args.source))
    if image is None:
        raise RuntimeError(f"Could not load {args.source}")
    pipeline = build_pipeline(
        args.model,
        args.output_dir,
        source_name=args.source.stem,
        classes=args.classes,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        max_det=args.max_det,
        crop=args.crop,
        background=args.background,
    )
    pipeline.validate()
    pipeline.describe()
    paths = pipeline(image)
    print(f"Saved {len(paths)} objects to {args.output_dir}")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
