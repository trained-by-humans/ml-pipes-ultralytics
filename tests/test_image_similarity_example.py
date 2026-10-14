"""Test image similarity with synthetic embeddings, without real inference."""
from __future__ import annotations

import re
import runpy
import sys
from pathlib import Path
from unittest.mock import Mock, create_autospec

import cv2
import numpy as np
import pytest
import torch
from ultralytics.engine.model import Model

from ml_pipes.ultralytics import yolo

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def example() -> dict:
    return runpy.run_path(str(ROOT / "examples" / "run_image_similarity.py"))


@pytest.fixture
def native_model(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock]:
    model = create_autospec(Model, instance=True, spec_set=True)
    model.embed.return_value = [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])]
    factory = create_autospec(yolo.YOLO, spec_set=True, return_value=model)
    monkeypatch.setattr(yolo, "YOLO", factory)
    return factory, model


@pytest.mark.parametrize(
    "first,second,expected",
    [
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 1.0),
        ([1.0, 2.0, 3.0], [10.0, 20.0, 30.0], 1.0),
        ([1.0, 0.0], [0.0, 1.0], 0.0),
        ([1.0, 0.0], [-1.0, 0.0], -1.0),
        ([1.0, 0.0], [3.0, 4.0], 0.6),
        ([1e-10, 0.0], [2e-10, 0.0], 1.0),
    ],
    ids=["identical", "scaled", "orthogonal", "opposite", "partial", "small-nonzero"],
)
def test_cosine_similarity(
    example: dict, first: list[float], second: list[float], expected: float
) -> None:
    embeddings = [torch.tensor(first), torch.tensor(second)]
    originals = [vector.clone() for vector in embeddings]

    score = example["CosineSimilarity"]()(embeddings)

    assert type(score) is float
    assert -1.0 <= score <= 1.0
    assert score == pytest.approx(expected, abs=1e-6)
    for vector, original in zip(embeddings, originals):
        torch.testing.assert_close(vector, original)


def test_cosine_calculation_uses_pytorch(
    example: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    cosine = Mock(return_value=torch.tensor(0.75))
    monkeypatch.setattr(torch.nn.functional, "cosine_similarity", cosine)
    embeddings = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]

    assert example["CosineSimilarity"]()(embeddings) == 0.75

    cosine.assert_called_once()
    for actual, expected in zip(cosine.call_args.args, embeddings):
        torch.testing.assert_close(actual, expected)
    assert cosine.call_args.kwargs == {"dim": 0, "eps": 0.0}


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_native_floating_dtypes_are_supported(example: dict, dtype: torch.dtype) -> None:
    score = example["CosineSimilarity"]()(
        [torch.tensor([1.0, 0.0], dtype=dtype), torch.tensor([0.0, 1.0], dtype=dtype)]
    )
    assert score == 0.0


@pytest.mark.parametrize(
    "embeddings,message",
    [
        ([], "exactly two"),
        ([torch.ones(2)], "exactly two"),
        ([torch.ones(2)] * 3, "exactly two"),
        ([torch.empty(0), torch.empty(0)], "non-empty"),
        ([torch.ones(1, 2), torch.ones(1, 2)], "one-dimensional"),
        ([torch.tensor(1.0), torch.tensor(1.0)], "one-dimensional"),
        ([torch.ones(2), torch.ones(3)], "same shape"),
        ([torch.zeros(2), torch.ones(2)], "non-zero norms"),
        ([torch.tensor([float("nan"), 1.0]), torch.ones(2)], "finite values"),
        ([torch.ones(2), torch.tensor([float("inf"), 1.0])], "finite values"),
        ([torch.full((2,), 3e38), torch.ones(2)], "finite, non-zero norms"),
    ],
)
def test_invalid_embedding_pairs_fail_clearly(
    example: dict, embeddings: list[torch.Tensor], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        example["CosineSimilarity"]()(embeddings)


def test_pipeline_uses_embed_adapter_and_reuses_one_model(
    example: dict, native_model: tuple[Mock, Mock], tmp_path: Path
) -> None:
    factory, model = native_model
    model.embed.side_effect = [
        [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])],
        [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])],
    ]
    pipeline = example["build_pipeline"]("custom.pt", imgsz=320, device="cpu")
    pipeline.validate()
    paths = [str(tmp_path / name) for name in ("first.jpg", "second.jpg", "third.jpg", "fourth.jpg")]
    for path in paths:
        Path(path).write_bytes(b"Mocked image")

    first_score = pipeline(paths[:2])
    second_score = pipeline(paths[2:])
    assert type(first_score) is float and type(second_score) is float
    assert first_score == 1.0
    assert second_score == 0.0

    factory.assert_called_once_with(model="custom.pt", task=None, verbose=False)
    assert [call.kwargs for call in model.embed.call_args_list] == [
        {"source": pair, "imgsz": 320, "device": "cpu", "rect": False}
        for pair in (paths[:2], paths[2:])
    ]
    model.assert_not_called()
    model.predict.assert_not_called()
    model.track.assert_not_called()


@pytest.mark.parametrize("count", [0, 1, 3])
def test_pipeline_requires_two_images_before_embedding(
    example: dict, native_model: tuple[Mock, Mock], tmp_path: Path, count: int
) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"Mocked image")
    pipeline = example["build_pipeline"]()

    with pytest.raises(ValueError, match="exactly two image paths"):
        pipeline([str(image)] * count)

    native_model[1].embed.assert_not_called()


@pytest.mark.parametrize("unavailable", ["missing.jpg", "directory"])
def test_pipeline_requires_both_files_before_embedding(
    example: dict, native_model: tuple[Mock, Mock], tmp_path: Path, unavailable: str
) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"Mocked image")
    (tmp_path / "directory").mkdir()
    pipeline = example["build_pipeline"]()

    with pytest.raises(ValueError, match="Image file does not exist"):
        pipeline([str(image), str(tmp_path / unavailable)])

    native_model[1].embed.assert_not_called()


@pytest.mark.parametrize("custom_inputs", [False, True])
@pytest.mark.parametrize("second_vector,expected_score", [([1.0, 0.0], 1.0), ([3.0, 4.0], 0.6)])
def test_cli_defaults_and_custom_inputs_print_similarity(
    example: dict,
    native_model: tuple[Mock, Mock],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    custom_inputs: bool,
    second_vector: list[float],
    expected_score: float,
) -> None:
    factory, model = native_model
    model.embed.return_value = [torch.tensor([1.0, 0.0]), torch.tensor(second_vector)]
    monkeypatch.setitem(example["main"].__globals__, "SAMPLE_IMAGES", tmp_path)
    names = ("first.png", "second.png") if custom_inputs else ("bus-1.png", "bus-2.png")
    paths = [tmp_path / name for name in names]
    for path in paths:
        path.write_bytes(b"Mocked inference does not decode these files.")
    argv = ["run_image_similarity.py"]
    if custom_inputs:
        argv += [
            "--image-a", str(paths[0]), "--image-b", str(paths[1]),
            "--model", "custom.pt", "--imgsz", "320", "--device", "cpu",
        ]
    monkeypatch.setattr(sys, "argv", argv)

    assert example["main"]() == 0

    assert capsys.readouterr().out.splitlines() == [
        f"Image A: {names[0]}",
        f"Image B: {names[1]}",
        f"Cosine similarity: {expected_score:.6f}",
    ]
    factory.assert_called_once_with(
        model="custom.pt" if custom_inputs else "yolo26n.pt", task=None, verbose=False
    )
    model.embed.assert_called_once_with(
        source=[str(path) for path in paths],
        imgsz=320 if custom_inputs else 640,
        device="cpu" if custom_inputs else None,
        rect=False,
    )


@pytest.mark.parametrize(
    "arguments,message",
    [
        (["--threshold", "0.9"], "unrecognized arguments"),
        (["--imgsz", "0"], "must be greater than zero"),
        (["--imgsz", "-1"], "must be greater than zero"),
        (["--image-a", "missing.jpg"], "Image file does not exist"),
        (["--image-b", "missing.jpg"], "Image file does not exist"),
    ],
)
def test_invalid_cli_arguments_fail_before_model_construction(
    example: dict,
    native_model: tuple[Mock, Mock],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    message: str,
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_image_similarity.py", *arguments])

    with pytest.raises(SystemExit) as raised:
        example["main"]()

    assert raised.value.code == 2
    assert message in capsys.readouterr().err
    native_model[0].assert_not_called()


def test_native_input_errors_propagate(
    example: dict, native_model: tuple[Mock, Mock], tmp_path: Path
) -> None:
    error = ValueError("Native image decoding failed")
    native_model[1].embed.side_effect = error
    pipeline = example["build_pipeline"]()
    paths = [str(tmp_path / name) for name in ("first.jpg", "second.jpg")]
    for path in paths:
        Path(path).write_bytes(b"Invalid image data")

    with pytest.raises(ValueError) as raised:
        pipeline(paths)

    assert raised.value is error


def test_tutorial_snippets_run_in_sequence(
    native_model: tuple[Mock, Mock], capsys: pytest.CaptureFixture[str]
) -> None:
    tutorial = ROOT / "docs" / "tutorials" / "image_similarity.md"
    snippets = re.findall(r"```python\n(.*?)\n```", tutorial.read_text(), re.DOTALL)
    assert len(snippets) == 4
    namespace = {"__name__": "image_similarity_tutorial"}
    native_model[1].embed.side_effect = [
        [torch.tensor([1.0, 0.0])],
        [torch.tensor([1.0, 0.0]), torch.tensor([3.0, 4.0])],
        [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])],
    ]

    for snippet in snippets:
        exec(compile(snippet, str(tutorial), "exec", dont_inherit=True), namespace)

    torch.testing.assert_close(namespace["embedding"], torch.tensor([1.0, 0.0]))
    assert len(namespace["embedding_pipeline"].operators) == 1
    assert len(namespace["pipeline"].operators) == 2
    assert type(namespace["similarity"]) is float
    assert namespace["similarity"] == pytest.approx(0.6)
    assert namespace["flip_similarity"] == 1.0
    assert capsys.readouterr().out.splitlines() == [
        "Image: bus-1.png",
        "Embedding shape: torch.Size([2])",
        "First 8 values: [1.0, 0.0]",
        "Image A: bus-1.png",
        "Image B: bus-2.png",
        "Cosine similarity: 0.600000",
        "Image A: bus-1.png",
        "Image B: bus-1-flipped.png",
        "Cosine similarity: 1.000000",
    ]
    assert native_model[0].call_count == 2
    assert all(
        call.kwargs == {"model": "yolo26n.pt", "task": None, "verbose": False}
        for call in native_model[0].call_args_list
    )
    assert [call.kwargs for call in native_model[1].embed.call_args_list] == [
        {"source": str(namespace["image_a"]), "imgsz": 640, "device": "cpu", "rect": False},
    ] + [
        {"source": [str(namespace["image_a"]), str(other)], "imgsz": 640, "device": "cpu", "rect": False}
        for other in (namespace["image_b"], namespace["flipped_image"])
    ]


def test_tutorial_assets_share_a_width_and_include_a_pixel_exact_flip() -> None:
    images = ROOT / "docs" / "assets" / "image_similarity"
    first = cv2.imread(str(images / "bus-1.png"))
    second = cv2.imread(str(images / "bus-2.png"))
    flipped = cv2.imread(str(images / "bus-1-flipped.png"))

    assert first is not None and second is not None and flipped is not None
    assert first.shape[1] == second.shape[1] == flipped.shape[1] == 1112
    np.testing.assert_array_equal(flipped, cv2.flip(first, 1))
