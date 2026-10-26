"""Detect objects and classify each crop with a second YOLO model.

Run from the repository root with an image of your choice:
    python examples/run_detect_and_classify.py
    python examples/run_detect_and_classify.py --source image.jpg --show

Defaults use the tutorial's eight-dog image, a COCO detector, and an ImageNet
classifier. Rank only ImageNet's 118 domestic dog breed classes, retaining the
original probabilities without renormalizing. Boxes show the highest-scoring
breed, adding the runner-up as first / second when the top score is below 50%.
Scores are not a guarantee of breed identity and are separate from detection
confidence.

For bears with the same stock classifier:
    python examples/run_detect_and_classify.py --source bears.jpg --class-name bear \
        --classification-classes 294 295 296 297 387 388
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from PIL.Image import Image
from ultralytics.engine.results import Results
from ultralytics.utils.plotting import save_one_box

from common import EXAMPLE_ASSETS
from ml_pipes.core import Pipeline
from ml_pipes.operator import Operator
from ml_pipes.standard import Gather, Pick, Recall, Scatter, Select, Store
from ml_pipes.ultralytics import results, yolo

ImageArray = npt.NDArray[np.uint8]
# Stock ImageNet IDs: Chihuahua through Mexican hairless dog, excluding wild canids.
DOG_BREED_CLASSES = tuple(range(151, 269))
TUTORIAL_IMAGE = Path(__file__).resolve().parents[1] / "docs/assets/detect_and_classify/dogs.png"


@Operator
class FilterResults:
    """Keep exact class-name matches as a native Results object."""

    def __init__(self, class_name: str) -> None:
        self.class_name = class_name

    def __call__(self, result: Results) -> Results:
        if self.class_name not in result.names.values():
            raise ValueError(f"Detector has no class named {self.class_name!r}.")
        if result.boxes is None:
            empty = result.new()
            empty.update(boxes=np.empty((0, 6), dtype=np.float32))
            return empty
        indices = [
            index
            for index, class_id in enumerate(result.boxes.cls.tolist())
            if result.names[int(class_id)] == self.class_name
        ]
        return result[indices]


@Operator
class CropResults:
    """Return aligned detections and in-memory crops using Ultralytics."""

    def __init__(self, *, save: bool = False, BGR: bool = True, gain: float = 1.0, pad: int = 0) -> None:
        self.save = save
        self.BGR = BGR
        self.gain = gain
        self.pad = pad

    def __call__(self, result: Results) -> tuple[Results, list[ImageArray]]:
        indices, crops = [], []
        for index, box in enumerate(result.boxes.cpu().numpy()):
            if not np.isfinite(box.xyxy).all():
                warnings.warn(f"Skipping detection {index}: non-finite bounding box.", RuntimeWarning)
                continue
            crop = save_one_box(
                box.xyxy, result.orig_img,
                save=self.save, BGR=self.BGR, gain=self.gain, pad=self.pad,
            )
            if not crop.size:
                warnings.warn(f"Skipping detection {index}: empty bounding-box crop.", RuntimeWarning)
                continue
            indices.append(index)
            crops.append(crop.copy())
        return result[indices], crops


@Operator
class FilterClasses:
    """Keep candidate classification classes with their original scores."""

    def __init__(self, class_ids: tuple[int, ...] | None = None) -> None:
        if class_ids is not None and not class_ids:
            raise ValueError("Select at least one classification class ID.")
        self.class_ids = class_ids

    def __call__(self, result: Results) -> Results:
        if result.probs is None:
            raise ValueError("Classifier returned no probabilities; use a classification (-cls) checkpoint.")
        class_ids = tuple(result.names) if self.class_ids is None else self.class_ids
        if not class_ids or any(
            class_id not in result.names or not 0 <= class_id < len(result.probs.data)
            for class_id in class_ids
        ):
            raise ValueError("Classifier lacks the requested class IDs; choose IDs that match its checkpoint.")
        filtered = result.new()
        # Native classification IDs index the score vector, so remap names too.
        filtered.names = {index: result.names[class_id] for index, class_id in enumerate(class_ids)}
        filtered.update(probs=result.probs.data[list(class_ids)])
        return filtered


@Operator
class LabelDetections:
    """Build a display-only Results with a different classification label per box."""

    def __call__(self, detections: Results, classifications: list[Results]) -> Results:
        if len(detections.boxes) != len(classifications):
            raise ValueError("Each detection must have one classification result.")
        # Native Plot looks up names[box.cls]. Give every box its own display ID;
        # changing names[dog_id] alone would give all dogs the same breed label.
        boxes = detections.boxes.cpu().numpy().data[:, [0, 1, 2, 3, -2, -1]].copy()
        boxes[:, -1] = np.arange(len(boxes))
        labeled = detections.new()
        labeled.names = {
            index: self._classification_label(result)
            for index, result in enumerate(classifications)
        }
        labeled.update(boxes=boxes)
        return labeled

    @staticmethod
    def _classification_label(result: Results) -> str:
        """Show a runner-up only when the highest classification score is below 50%."""
        count = 2 if float(result.probs.top1conf) < 0.5 else 1
        return " / ".join(result.names[class_id].replace('_', ' ') for class_id in result.probs.top5[:count])


def build_pipeline(
    det_model: yolo.Model = "yolo26n.pt",
    cls_model: yolo.Model = "yolo26n-cls.pt",
    *,
    output: Path,
    class_name: str = "dog",
    classification_classes: tuple[int, ...] | None = DOG_BREED_CLASSES,
    det_imgsz: int = 960,
    cls_imgsz: int = 224,
    conf: float = 0.25,
    iou: float = 0.7,
    device: str | None = None,
    show: bool = False,
) -> Pipeline[ImageArray, ImageArray | Image]:
    """Reuse two models, with one classifier invocation per non-empty crop."""
    return Pipeline(
        [
            yolo.Predict(model=det_model, conf=conf, iou=iou, imgsz=det_imgsz, device=device),
            Select(0),
            FilterResults(class_name),
            CropResults(),
            Store("detections", source=0),
            Pick(1),  # Pass only the crops to the classifier.
            Scatter(max_concurrency=1),
            yolo.Predict(model=cls_model, imgsz=cls_imgsz, device=device),
            Select(0),
            FilterClasses(classification_classes),
            Gather(),
            Recall("detections", prepend=True),
            LabelDetections(),
            results.Plot(
                conf=False, line_width=2, pil=True, font_size=18,
                show=show, save=True, filename=str(output),
            ),
        ],
        auto_validate=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=TUTORIAL_IMAGE, help="Input image (default: tutorial's eight dogs).")
    parser.add_argument("--det-model", default="yolo26n.pt")
    parser.add_argument("--cls-model", default="yolo26n-cls.pt")
    parser.add_argument("--class-name", default="dog", help="Exact detector class label to crop.")
    parser.add_argument(
        "--classification-classes", type=int, nargs="+", default=DOG_BREED_CLASSES,
        help="Classifier class IDs to rank (default: ImageNet dog breeds 151-268).",
    )
    parser.add_argument("--det-imgsz", type=int, default=960)
    parser.add_argument("--cls-imgsz", type=int, default=224)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=EXAMPLE_ASSETS / "detect-and-classify")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    image = cv2.imread(str(args.source))
    if image is None:
        parser.error(f"Could not load {args.source}")
    annotated_path = args.output_dir / "annotated.jpg"
    if args.source.resolve() == annotated_path.resolve():
        parser.error("Output would overwrite the source image; choose a different --output-dir.")
    classification_classes = tuple(args.classification_classes)
    pipeline = build_pipeline(
        args.det_model, args.cls_model, output=annotated_path, class_name=args.class_name,
        classification_classes=classification_classes,
        det_imgsz=args.det_imgsz, cls_imgsz=args.cls_imgsz, conf=args.conf, iou=args.iou, device=args.device,
        show=args.show,
    )
    pipeline.validate()
    pipeline.describe()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline(image)
    print(f"Saved {annotated_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
