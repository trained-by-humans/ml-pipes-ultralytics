"""Run YOLO detection on video through an explicit OpenCV capture loop.

Run from the repository root:
    python examples/run_detection_video.py --input path/to/video.mp4 --show
    python examples/run_detection_video.py --input 0 --show
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import numpy.typing as npt

from ml_pipes.core import Pipeline
from ml_pipes.operator import Operator
from ml_pipes.standard import Select
from ml_pipes.ultralytics import results, yolo


def build_frame_pipeline(
    model: str,
    *,
    classes: list[int] | None,
    conf: float,
    iou: float,
    imgsz: int,
    device: str | None,
    max_det: int,
) -> Pipeline[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Build one non-streaming inference and annotation step for each BGR frame."""
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
            results.Plot(),
        ],
        auto_validate=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Video path or camera index, such as 0.")
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--classes", type=int, nargs="+", default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--show", action="store_true", help="Display frames; press q or Escape to stop.")
    parser.add_argument("--output", type=Path, default=None, help="Optional annotated MP4 output path.")
    args = parser.parse_args()

    pipeline = build_frame_pipeline(
        args.model,
        classes=args.classes,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        max_det=args.max_det,
    )
    pipeline.validate()
    pipeline.describe()

    source: str | int = int(args.input) if args.input.isdecimal() else args.input
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.input}")

    video_writer: cv2.VideoWriter | None = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        video_writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not video_writer.isOpened():
            raise RuntimeError(f"Could not create {args.output}")

    frames = 0
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            annotated = pipeline(frame)
            frames += 1
            if video_writer is not None:
                video_writer.write(annotated)
            if args.show:
                cv2.imshow("ml-pipes Ultralytics", annotated)
                if cv2.waitKey(1) & 0xFF in {27, ord("q")}:
                    break
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"Processed {frames} frames")
    if args.output is not None:
        print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
