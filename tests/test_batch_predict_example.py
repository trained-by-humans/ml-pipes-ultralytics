"""Exercise the runnable tutorial with local images and a mocked YOLO model."""
from __future__ import annotations

import argparse
import runpy
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, create_autospec
from zipfile import ZipFile

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


def test_default_archive_extracts_only_100_distinct_images_and_reuses_cache(
    example: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_bytes = BytesIO()
    with ZipFile(archive_bytes, "w") as archive:
        for index in range(128):
            archive.writestr(f"coco128/images/train2017/{index:03}.jpg", f"image-{index}")
        archive.writestr("coco128/labels/train2017/000.txt", "0 0.5 0.5 1 1")

    response = Mock()
    response.__enter__ = Mock(return_value=BytesIO(archive_bytes.getvalue()))
    response.__exit__ = Mock(return_value=False)
    download = Mock(return_value=response)
    globals_ = example["default_archive"].__globals__
    monkeypatch.setitem(globals_, "EXAMPLE_ASSETS", tmp_path)
    monkeypatch.setitem(globals_, "urlopen", download)

    directory = example["default_archive"]()

    assert sorted(path.name for path in directory.iterdir()) == [f"{index:03}.jpg" for index in range(100)]
    assert (directory / "099.jpg").read_bytes() == b"image-99"
    assert example["default_archive"]() == directory
    download.assert_called_once_with(
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip", timeout=60
    )


@pytest.mark.parametrize("value", ["0", "-1"])
def test_batch_size_and_worker_count_must_be_positive(example: dict, value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="greater than zero"):
        example["positive_int"](value)


def test_empty_archive_fails_before_model_construction(
    example: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    factory = Mock()
    monkeypatch.setattr(yolo, "YOLO", factory)
    monkeypatch.setattr(sys, "argv", ["run_batch_predict.py", "--source", str(tmp_path)])

    with pytest.raises(SystemExit) as raised:
        example["main"]()

    assert raised.value.code == 2
    assert "No images matching" in capsys.readouterr().err
    factory.assert_not_called()
