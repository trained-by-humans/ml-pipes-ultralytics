"""Native-shaped Ultralytics YOLO operators for :mod:`ml_pipes`."""
from __future__ import annotations

from typing import Any

from ml_pipes.inspection import TextBlock
from ml_pipes.inspection._global_registry import register_value_formatter

from .yolo import YOLOEmbed, YOLOPredict, YOLOTrack
from . import result


def _format_result(value: Any) -> list[TextBlock]:
    fields = ("boxes", "masks", "probs", "keypoints", "obb", "semantic_mask", "depth")
    summary = value.verbose().strip() if any(getattr(value, field, None) is not None for field in fields) else "no predictions"
    return [TextBlock("Ultralytics Results", [
        ("path", str(value.path)), ("original shape", str(value.orig_shape)),
        ("detections", str(0 if value.boxes is None else len(value.boxes))),
        ("masks", str(0 if value.masks is None else len(value.masks))),
        ("keypoints", str(0 if value.keypoints is None else len(value.keypoints))),
        ("oriented boxes", str(0 if value.obb is None else len(value.obb))),
        ("classification", "yes" if value.probs is not None else "no"),
        ("semantic mask", "yes" if getattr(value, "semantic_mask", None) is not None else "no"),
        ("depth map", "yes" if getattr(value, "depth", None) is not None else "no"),
        ("summary", summary),
    ])]


try:
    from ultralytics.engine.results import Results as _UltralyticsResults
except ImportError:  # pragma: no cover
    pass
else:
    register_value_formatter(_UltralyticsResults, _format_result)


__all__ = [
    "result",
    "YOLOEmbed",
    "YOLOPredict",
    "YOLOTrack",
]
