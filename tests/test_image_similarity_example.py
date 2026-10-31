"""Test image similarity with synthetic embeddings, without real inference."""
from __future__ import annotations

import runpy
from pathlib import Path
from unittest.mock import Mock, create_autospec

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


def test_pipeline_converts_native_embeddings_to_similarity(
    example: dict, native_model: tuple[Mock, Mock], tmp_path: Path
) -> None:
    _, model = native_model
    model.embed.return_value = [torch.tensor([1.0, 0.0]), torch.tensor([3.0, 4.0])]
    pipeline = example["build_pipeline"]("custom.pt", imgsz=320, device="cpu")
    pipeline.validate()
    paths = [str(tmp_path / name) for name in ("first.jpg", "second.jpg")]
    for path in paths:
        Path(path).write_bytes(b"Mocked image")

    score = pipeline(paths)

    assert type(score) is float
    assert score == pytest.approx(0.6)
    # Matching whole-image preprocessing is part of this comparison workflow.
    model.embed.assert_called_once_with(source=paths, imgsz=320, device="cpu", rect=False)


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
