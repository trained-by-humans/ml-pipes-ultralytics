"""Pipeline operators for configurable native :class:`ultralytics.Results` methods."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np
import numpy.typing as npt
from torch import Tensor
from ultralytics.engine.results import Results

from ml_pipes.operator import Operator
from ml_pipes.standard import SideEffectOp

if TYPE_CHECKING:
    from pandas import DataFrame

__all__ = ["Plot", "Save", "SaveCrop", "SaveTXT", "Show", "Summary", "To", "ToCSV", "ToDataFrame", "ToJSON"]

ResultT = TypeVar("ResultT", bound=Results)


@Operator
class To:
    """Call :meth:`Results.to` with device or dtype configuration."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def __call__(self, results: Results) -> Results:
        return results.to(*self.args, **self.kwargs)


@Operator
class Plot:
    """Render a native result with :meth:`Results.plot`."""

    def __init__(
        self,
        conf: bool = True,
        line_width: float | None = None,
        font_size: float | None = None,
        font: str = "Arial.ttf",
        pil: bool = False,
        img: npt.NDArray[np.generic] | Tensor | None = None,
        kpt_radius: int = 5,
        kpt_line: bool = True,
        labels: bool = True,
        boxes: bool = True,
        masks: bool = True,
        probs: bool = True,
        show: bool = False,
        save: bool = False,
        filename: str | None = None,
        color_mode: str = "class",
        txt_color: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        self.conf = conf
        self.line_width = line_width
        self.font_size = font_size
        self.font = font
        self.pil = pil
        self.img = img
        self.kpt_radius = kpt_radius
        self.kpt_line = kpt_line
        self.labels = labels
        self.boxes = boxes
        self.masks = masks
        self.probs = probs
        self.show = show
        self.save = save
        self.filename = filename
        self.color_mode = color_mode
        self.txt_color = txt_color

    def __call__(self, results: Results) -> npt.NDArray[np.uint8]:
        return results.plot(
            conf=self.conf,
            line_width=self.line_width,
            font_size=self.font_size,
            font=self.font,
            pil=self.pil,
            img=self.img,
            kpt_radius=self.kpt_radius,
            kpt_line=self.kpt_line,
            labels=self.labels,
            boxes=self.boxes,
            masks=self.masks,
            probs=self.probs,
            show=self.show,
            save=self.save,
            filename=self.filename,
            color_mode=self.color_mode,
            txt_color=self.txt_color,
        )


@Operator
class Show(SideEffectOp[ResultT], Generic[ResultT]):
    """Display a native result using :meth:`Results.show`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def effect(self, payload: ResultT) -> None:
        payload.show(*self.args, **self.kwargs)


@Operator
class ToDataFrame:
    """Convert a native result to a pandas dataframe."""

    def __init__(self, normalize: bool = False, decimals: int = 5) -> None:
        self.normalize = normalize
        self.decimals = decimals

    def __call__(self, results: Results) -> DataFrame:
        return results.to_df(normalize=self.normalize, decimals=self.decimals)


@Operator
class ToCSV:
    """Convert a native result to CSV."""

    def __init__(self, normalize: bool = False, decimals: int = 5) -> None:
        self.normalize = normalize
        self.decimals = decimals

    def __call__(self, results: Results) -> str:
        return results.to_csv(normalize=self.normalize, decimals=self.decimals)


@Operator
class ToJSON:
    """Convert a native result to JSON."""

    def __init__(self, normalize: bool = False, decimals: int = 5) -> None:
        self.normalize = normalize
        self.decimals = decimals

    def __call__(self, results: Results) -> str:
        return results.to_json(normalize=self.normalize, decimals=self.decimals)


@Operator
class SaveTXT(SideEffectOp[ResultT], Generic[ResultT]):
    """Save a native result's labels with :meth:`Results.save_txt`."""

    def __init__(self, txt_file: str | Path, save_conf: bool = False) -> None:
        self.txt_file = txt_file
        self.save_conf = save_conf

    def effect(self, payload: ResultT) -> None:
        payload.save_txt(self.txt_file, save_conf=self.save_conf)


@Operator
class SaveCrop(SideEffectOp[ResultT], Generic[ResultT]):
    """Save crops from a native result with :meth:`Results.save_crop`."""

    def __init__(self, save_dir: str | Path, file_name: str | Path = Path("im.jpg")) -> None:
        self.save_dir = save_dir
        self.file_name = file_name

    def effect(self, payload: ResultT) -> None:
        payload.save_crop(self.save_dir, file_name=self.file_name)


@Operator
class Save(SideEffectOp[ResultT], Generic[ResultT]):
    """Save an annotated native result with :meth:`Results.save`."""

    def __init__(self, filename: str | None = None, *args: Any, **kwargs: Any) -> None:
        self.filename = filename
        self.args = args
        self.kwargs = kwargs

    def effect(self, payload: ResultT) -> None:
        payload.save(self.filename, *self.args, **self.kwargs)


@Operator
class Summary:
    """Summarize a native result as dictionaries."""

    def __init__(self, normalize: bool = False, decimals: int = 5) -> None:
        self.normalize = normalize
        self.decimals = decimals

    def __call__(self, results: Results) -> list[dict[str, Any]]:
        return results.summary(normalize=self.normalize, decimals=self.decimals)
