"""Native-shaped, non-streaming Ultralytics YOLO operators."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt
from PIL.Image import Image
from torch import Tensor
from ultralytics import YOLO
from ultralytics.engine.model import Model as NativeModel
from ultralytics.engine.results import Results

from ml_pipes.operator import Operator

Source: TypeAlias = str | Path | int | Image | npt.NDArray[np.generic] | Tensor
Sources: TypeAlias = Source | list[Source] | tuple[Source, ...]
Model: TypeAlias = str | Path | NativeModel
Device: TypeAlias = str | int | Sequence[int]
Classes: TypeAlias = int | Sequence[int]

__all__ = ["Classes", "Device", "Embed", "Predict", "Source", "Sources", "Track"]


class _YOLOOperation:
    """Shared setup for an operator that owns one native YOLO model."""

    def __init__(
        self,
        model: Model,
        task: str | None,
        verbose: bool,
        **options: Any,
    ) -> None:
        _reject_streaming(options)
        self.model = _resolve_model(model, task=task, verbose=verbose)
        self.options = dict(options)


@Operator
class Predict(_YOLOOperation):
    """A non-streaming pipeline boundary around :meth:`ultralytics.YOLO.predict`."""

    def __init__(
        self,
        model: Model = "yolo26n.pt",
        task: str | None = None,
        verbose: bool = False,
        **predict_options: Any,
    ) -> None:
        _reject_embed(predict_options, operation="Predict")
        super().__init__(model, task, verbose, **predict_options)

    def __call__(self, source: Sources) -> list[Results]:
        return self.model(source=source, **self.options)


@Operator
class Embed(_YOLOOperation):
    """A non-streaming pipeline boundary around :meth:`ultralytics.YOLO.embed`."""

    def __init__(
        self,
        model: Model = "yolo26n.pt",
        task: str | None = None,
        verbose: bool = False,
        **embed_options: Any,
    ) -> None:
        super().__init__(model, task, verbose, **embed_options)

    def __call__(self, source: Sources) -> list[Tensor]:
        return self.model.embed(source=source, **self.options)


@Operator
class Track(_YOLOOperation):
    """A non-streaming pipeline boundary around :meth:`ultralytics.YOLO.track`."""

    def __init__(
        self,
        model: Model = "yolo26n.pt",
        task: str | None = None,
        verbose: bool = False,
        *,
        persist: bool = False,
        **track_options: Any,
    ) -> None:
        if not isinstance(persist, bool):
            raise TypeError("persist must be bool.")
        _reject_embed(track_options, operation="Track")
        super().__init__(model, task, verbose, **track_options)
        self.persist = persist

    def __call__(self, source: Sources) -> list[Results]:
        return self.model.track(source=source, persist=self.persist, **self.options)


def _reject_streaming(options: dict[str, Any]) -> None:
    if options.get("stream", False):
        raise ValueError("Streaming is not supported by ml-pipes-ultralytics yet.")


def _resolve_model(model: Model, *, task: str | None, verbose: bool) -> NativeModel:
    """Reuse a supplied native model, or construct one from a model specification."""
    if isinstance(model, NativeModel):
        return model
    return YOLO(model=model, task=task, verbose=verbose)


def _reject_embed(options: dict[str, Any], *, operation: str) -> None:
    if "embed" in options:
        raise TypeError(f"{operation} does not accept embed; use yolo.Embed instead.")
