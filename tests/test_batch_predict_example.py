"""Exercise the runnable tutorial with local images and a mocked YOLO model."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path
from unittest.mock import Mock, create_autospec

import cv2
import numpy as np
import pytest
from ultralytics.engine.model import Model
from ultralytics.engine.results import Results

from ml_pipes.ultralytics import yolo

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def example(monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.syspath_prepend(str(EXAMPLES))
    return runpy.run_path(str(EXAMPLES / "run_batch_predict.py"))


@pytest.mark.parametrize("mode,call_sizes", [("single", [1, 1, 1]), ("batch", [2, 1]), ("concurrent", [2, 1])])
def test_cli_modes_preserve_images_order_and_partial_batch(
    mode: str,
    call_sizes: list[int],
    example: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    images = []
    # Write in reverse order to check the script's sorted archive traversal.
    for index in reversed(range(3)):
        image = np.empty((17 + index * 2, 23 + index * 2, 3), dtype=np.uint8)
        image[:] = (10 + index, 20 + index, 30 + index)
        assert cv2.imwrite(str(tmp_path / f"{index}.png"), image)
        images.insert(0, image)

    model = create_autospec(Model, instance=True, spec_set=True)

    def predict(source, **options):
        items = [source] if isinstance(source, str) else source
        arrays = [cv2.imread(item) if isinstance(item, str) else item for item in items]
        return [
            Results(
                orig_img=image,
                path="in-memory.jpg",
                names={0: f"image-{image[0, 0, 0]}"},
                boxes=np.array([[0, 0, 2, 2, 0.8, 0]], dtype=np.float32),
            )
            for image in arrays
        ]

    model.side_effect = predict
    factory = create_autospec(yolo.YOLO, spec_set=True, return_value=model)
    monkeypatch.setattr(yolo, "YOLO", factory)
    download = Mock(side_effect=AssertionError("Custom inputs must not download samples"))
    monkeypatch.setitem(example["main"].__globals__, "default_archive", download)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_batch_predict.py", "--source", str(tmp_path), "--pattern", "*.png",
            "--mode", mode, "--batch-size", "2", "--workers", "2",
            "--model", "custom.pt", "--conf", "0.4", "--imgsz", "320", "--device", "cpu",
        ],
    )

    assert example["main"]() == 0

    factory.assert_called_once_with(model="custom.pt", task=None, verbose=False)
    download.assert_not_called()
    calls = model.call_args_list
    batches = [
        [call.kwargs["source"]] if mode == "single" else call.kwargs["source"]
        for call in calls
    ]
    assert [len(batch) for batch in batches] == call_sizes
    for native_call in calls:
        assert {key: value for key, value in native_call.kwargs.items() if key != "source"} == {
            "conf": 0.4, "imgsz": 320, "device": "cpu",
        }
    inputs = [image for batch in batches for image in batch]
    if mode == "concurrent":
        for image, expected in zip(inputs, images):
            np.testing.assert_array_equal(image, expected)
    else:
        assert inputs == [str(tmp_path / f"{index}.png") for index in range(3)]
    summaries = [line for line in capsys.readouterr().out.splitlines() if ".png:" in line]
    assert summaries == [f"{index}.png: 1 image-{10 + index}," for index in range(3)]
