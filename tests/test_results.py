from __future__ import annotations

import numpy as np

from ml_pipes.ultralytics import result as result_ops


class FakeResults:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def to(self, *args: object, **kwargs: object) -> FakeResults:
        self.calls.append(("to", args, kwargs))
        return self

    def show(self, *args: object, **kwargs: object) -> None:
        self.calls.append(("show", args, kwargs))

    def plot(self, **kwargs: object) -> np.ndarray:
        self.calls.append(("plot", (), kwargs))
        return np.zeros((1, 1, 3), dtype=np.uint8)

    def to_df(self, **kwargs: object) -> str:
        self.calls.append(("to_df", (), kwargs))
        return "dataframe"

    def to_csv(self, *args: object, **kwargs: object) -> str:
        self.calls.append(("to_csv", args, kwargs))
        return "csv"

    def to_json(self, **kwargs: object) -> str:
        self.calls.append(("to_json", (), kwargs))
        return "json"

    def save_txt(self, *args: object, **kwargs: object) -> None:
        self.calls.append(("save_txt", args, kwargs))

    def save_crop(self, *args: object, **kwargs: object) -> None:
        self.calls.append(("save_crop", args, kwargs))

    def save(self, *args: object, **kwargs: object) -> str:
        self.calls.append(("save", args, kwargs))
        return "image.jpg"

    def summary(self, **kwargs: object) -> list[dict[str, object]]:
        self.calls.append(("summary", (), kwargs))
        return [{"class": 1}]


def test_result_operators_fix_configuration_at_construction() -> None:
    result = FakeResults()

    assert result_ops.To("cpu", non_blocking=True)(result) is result  # type: ignore[arg-type]
    assert result_ops.Show(False)(result) is result  # type: ignore[arg-type]
    assert result_ops.ToDataFrame(normalize=True, decimals=3)(result) == "dataframe"  # type: ignore[arg-type]
    assert result_ops.ToCSV(True, 3)(result) == "csv"  # type: ignore[arg-type]
    assert result_ops.ToJSON(normalize=True, decimals=3)(result) == "json"  # type: ignore[arg-type]
    assert result_ops.SaveTXT("labels.txt", save_conf=True)(result) is result  # type: ignore[arg-type]
    assert result_ops.SaveCrop("crops", "frame")(result) is result  # type: ignore[arg-type]
    assert result_ops.Save("image.jpg", False, line_width=2)(result) is result  # type: ignore[arg-type]
    assert result_ops.Summary(normalize=True, decimals=3)(result) == [{"class": 1}]  # type: ignore[arg-type]

    assert result.calls == [
        ("to", ("cpu",), {"non_blocking": True}),
        ("show", (False,), {}),
        ("to_df", (), {"normalize": True, "decimals": 3}),
        ("to_csv", (), {"normalize": True, "decimals": 3}),
        ("to_json", (), {"normalize": True, "decimals": 3}),
        ("save_txt", ("labels.txt",), {"save_conf": True}),
        ("save_crop", ("crops",), {"file_name": "frame"}),
        ("save", ("image.jpg", False), {"line_width": 2}),
        ("summary", (), {"normalize": True, "decimals": 3}),
    ]


def test_result_module_exposes_operators() -> None:
    assert result_ops.Plot.__name__ == "Plot"
    assert result_ops.SaveCrop.__name__ == "SaveCrop"
