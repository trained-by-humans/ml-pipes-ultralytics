from __future__ import annotations

import pytest
from torch import Tensor

from ml_pipes.core import Pipeline
from ml_pipes.ultralytics import YOLOEmbed, YOLOPredict, YOLOTrack


class FakeResults: pass


class FakeYOLOModel:
    def __init__(self, model: str = "yolo26n.pt", task: str | None = None, verbose: bool = False) -> None:
        self.model_name, self.task, self.verbose = model, task or "detect", verbose
        self.calls: list[tuple[str, object, dict[str, object]]] = []
    def __call__(self, source: object = None, **kwargs: object) -> list[FakeResults]:
        self.calls.append(("call", source, kwargs)); return [FakeResults()]
    def track(self, source: object, **kwargs: object) -> list[FakeResults]:
        self.calls.append(("track", source, kwargs)); return [FakeResults()]
    def embed(self, source: object = None, **kwargs: object) -> list[Tensor]:
        self.calls.append(("embed", source, kwargs)); return [Tensor()]


@pytest.fixture(autouse=True)
def native_yolo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ml_pipes.ultralytics.yolo.YOLO", FakeYOLOModel)
    monkeypatch.setattr("ml_pipes.ultralytics.yolo.NativeModel", FakeYOLOModel)


def test_native_yolo_forwards_native_source_and_options() -> None:
    operator = YOLOPredict("custom.pt", task="detect", verbose=True, conf=0.4)
    assert isinstance(operator("frame.jpg")[0], FakeResults)
    assert operator.model.calls == [("call", "frame.jpg", {"conf": 0.4})]


def test_operator_description_includes_the_configured_model() -> None:
    operator = YOLOPredict(model="custom.pt", conf=0.25)
    description = repr(Pipeline([operator], auto_validate=False))
    assert "YOLOPredict(model='custom.pt', conf=0.25)" in description
    assert "FakeYOLOModel object" not in description


def test_yolo_operator_classes_construct_their_own_native_model() -> None:
    assert isinstance(YOLOPredict("custom.pt"), YOLOPredict)
    assert isinstance(YOLOEmbed("custom.pt"), YOLOEmbed)
    assert isinstance(YOLOTrack("custom.pt"), YOLOTrack)


def test_yolo_operators_reuse_a_supplied_native_model() -> None:
    model = FakeYOLOModel("custom.pt")
    assert YOLOPredict(model).model is model
    assert YOLOEmbed(model).model is model
    assert YOLOTrack(model).model is model


def test_native_yolo_rejects_streaming() -> None:
    with pytest.raises(ValueError, match="Streaming"):
        YOLOPredict(stream=True)


def test_native_yolo_embed_is_a_dedicated_operator() -> None:
    embedder = YOLOEmbed("custom.pt", conf=0.4)
    embeddings = embedder("frame.jpg")
    assert isinstance(embeddings[0], Tensor)
    assert embedder.model.calls == [("embed", "frame.jpg", {"conf": 0.4})]
    with pytest.raises(TypeError, match="YOLOEmbed"):
        YOLOPredict(embed=[-2])


def test_native_yolo_call_accepts_data_only() -> None:
    with pytest.raises(TypeError):
        YOLOPredict()("frame.jpg", conf=0.4)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        YOLOPredict()()  # type: ignore[call-arg]


def test_native_yolo_track_forwards_options_and_persists_at_operator_scope() -> None:
    tracker = YOLOTrack("detect.pt", persist=True, tracker="bytetrack.yaml", conf=0.1)
    assert len(tracker("movie.mp4")) == 1
    assert tracker.model.calls == [("track", "movie.mp4", {"persist": True, "tracker": "bytetrack.yaml", "conf": 0.1})]


def test_native_yolo_track_rejects_streaming_when_pipeline_configuration_is_built() -> None:
    with pytest.raises(ValueError, match="Streaming"):
        YOLOTrack("detect.pt", stream=True)
