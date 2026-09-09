"""Run YOLO tracking on video through an explicit OpenCV capture loop.

Run from the repository root:
    python examples/run_track_objects.py --input path/to/video.mp4 --show
    python examples/run_track_objects.py --input 0 --show
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from ultralytics.engine.results import Results

from ml_pipes.core import Pipeline
from ml_pipes.operator import Operator
from ml_pipes.standard import Select
from ml_pipes.ultralytics import yolo


@Operator
class TrackTrace:
    """Plot native tracked results and retain a short trace for each track ID."""

    def __init__(self, length: int = 30) -> None:
        if length < 2:
            raise ValueError("length must be at least 2.")
        self.traces: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=length))

    def __call__(self, results: Results) -> npt.NDArray[np.uint8]:
        annotated = results.plot(color_mode="instance")
        if results.boxes is None or results.boxes.id is None:
            return annotated
        for box, track_id in zip(results.boxes.xywh.cpu().numpy(), results.boxes.id.int().cpu().tolist()):
            x, y = map(int, box[:2])
            trace = self.traces[track_id]
            trace.append((x, y))
            if len(trace) > 1:
                cv2.polylines(annotated, [np.asarray(trace, dtype=np.int32)], False, (230, 230, 230), 2)
        return annotated


def build_frame_pipeline(
    model: str,
    *,
    classes: list[int] | None,
    conf: float,
    iou: float,
    imgsz: int,
    device: str | None,
    max_det: int,
    tracker: str,
    trace_length: int,
) -> Pipeline[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Build one persistent tracking and rendering pipeline for BGR frames."""
    return Pipeline(
        [
            yolo.Track(
                model=model,
                persist=True,
                tracker=tracker,
                classes=classes,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                max_det=max_det,
            ),
            Select(0),
            TrackTrace(trace_length),
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
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--trace-length", type=int, default=30)
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
        tracker=args.tracker,
        trace_length=args.trace_length,
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
                cv2.imshow("ml-pipes Ultralytics tracking", annotated)
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
