from __future__ import annotations

from collections.abc import Sequence
from inspect import signature
from pathlib import Path
from unittest.mock import Mock, call, create_autospec, sentinel

import numpy as np
import numpy.typing as npt
import pytest
from PIL.Image import Image
from torch import Tensor
from ultralytics.engine.model import Model

from ml_pipes.core import Pipeline
from ml_pipes.standard import Gather, Scatter
from ml_pipes.ultralytics import yolo

OPERATORS = [yolo.Predict, yolo.Embed, yolo.Track]
METHODS = [(yolo.Predict, "__call__"), (yolo.Embed, "embed"), (yolo.Track, "track")]


@pytest.fixture(autouse=True)
def native_yolo(monkeypatch: pytest.MonkeyPatch) -> Mock:
    # Keep NativeModel real so the native-instance check is also exercised.
    factory = create_autospec(yolo.YOLO, spec_set=True)
    factory.return_value = create_autospec(Model, instance=True, spec_set=True)
    monkeypatch.setattr(yolo, "YOLO", factory)
    return factory


@pytest.mark.parametrize("operator_class", OPERATORS)
def test_default_model_configuration(operator_class, native_yolo: Mock) -> None:
    operator = operator_class()

    native_yolo.assert_called_once_with(model="yolo26n.pt", task=None, verbose=False)
    assert operator.model is native_yolo.return_value
    assert operator.options == {}
    if operator_class is yolo.Track:
        assert operator.persist is False


@pytest.mark.parametrize("operator_class", OPERATORS)
@pytest.mark.parametrize("model_spec", ["custom.pt", Path("custom.pt")], ids=["str", "path"])
def test_model_constructor_configuration_is_separate_from_call_options(
    operator_class, model_spec, native_yolo: Mock
) -> None:
    operator = operator_class(model_spec, task="segment", verbose=True, conf=0.4, imgsz=320)

    native_yolo.assert_called_once_with(model=model_spec, task="segment", verbose=True)
    assert operator.model is native_yolo.return_value
    assert operator.options == {"conf": 0.4, "imgsz": 320}


@pytest.mark.parametrize("operator_class", OPERATORS)
def test_supplied_native_model_is_reused_without_construction(
    operator_class, native_yolo: Mock
) -> None:
    model = native_yolo.return_value
    operator = operator_class(model, task="detect", verbose=True, conf=0.4)

    assert isinstance(model, Model)
    assert operator.model is model
    assert operator.options == {"conf": 0.4}
    operator("first.jpg")
    operator("second.jpg")
    assert operator.model is model
    native_yolo.assert_not_called()


@pytest.mark.parametrize("operator_class,method_name", METHODS)
def test_calls_forward_options_and_preserve_native_outputs_across_calls(
    operator_class, method_name: str, native_yolo: Mock
) -> None:
    options = {"conf": 0.0, "classes": [0, 2], "imgsz": 320, "stream": False}
    if operator_class is yolo.Embed:
        options["embed"] = [4, 6]
    extra = {"persist": True} if operator_class is yolo.Track else {}
    operator = operator_class("custom.pt", **extra, **options)
    model = operator.model
    method = model if method_name == "__call__" else getattr(model, method_name)
    outputs = [[sentinel.first, sentinel.second], []]
    method.side_effect = outputs

    assert operator("first.jpg") is outputs[0]
    assert operator("second.jpg") is outputs[1]
    assert operator.model is model
    assert operator.options == options
    assert method.call_args_list == [
        call(source="first.jpg", **extra, **options),
        call(source="second.jpg", **extra, **options),
    ]
    # Bind to the real API too: autospec's callable-object assertions can retain self.
    for native_call in method.call_args_list:
        signature(getattr(Model, method_name)).bind(model, *native_call.args, **native_call.kwargs)
    native_yolo.assert_called_once_with(model="custom.pt", task=None, verbose=False)
    for other_name in {"__call__", "embed", "track"} - {method_name}:
        other = model if other_name == "__call__" else getattr(model, other_name)
        other.assert_not_called()


@pytest.mark.parametrize("operator_class,method_name", METHODS)
def test_separately_constructed_operators_do_not_share_models_or_options(
    operator_class, method_name: str, native_yolo: Mock
) -> None:
    first_model = native_yolo.return_value
    second_model = create_autospec(Model, instance=True, spec_set=True)
    native_yolo.side_effect = [first_model, second_model]
    first = operator_class("custom.pt", conf=0.1)
    second = operator_class("custom.pt", conf=0.8)

    first("first.jpg")
    second("second.jpg")

    assert first.model is first_model
    assert second.model is second_model
    extra = {"persist": False} if operator_class is yolo.Track else {}
    first_method = first_model if method_name == "__call__" else getattr(first_model, method_name)
    second_method = second_model if method_name == "__call__" else getattr(second_model, method_name)
    assert first_method.call_args_list == [call(source="first.jpg", conf=0.1, **extra)]
    assert second_method.call_args_list == [call(source="second.jpg", conf=0.8, **extra)]
    assert native_yolo.call_count == 2


@pytest.mark.parametrize("operator_class", OPERATORS)
def test_streaming_is_rejected_before_model_construction(operator_class, native_yolo: Mock) -> None:
    with pytest.raises(ValueError, match="Streaming"):
        operator_class(stream=True)
    native_yolo.assert_not_called()


@pytest.mark.parametrize("operator_class", [yolo.Predict, yolo.Track])
@pytest.mark.parametrize("embed", [None, []])
def test_embedding_option_is_rejected_outside_embed(
    operator_class, embed, native_yolo: Mock
) -> None:
    with pytest.raises(TypeError, match="use yolo.Embed"):
        operator_class(embed=embed)
    native_yolo.assert_not_called()


@pytest.mark.parametrize("persist", [None, 0, 1, "true"])
def test_tracking_requires_boolean_persistence(persist, native_yolo: Mock) -> None:
    with pytest.raises(TypeError, match="persist must be bool"):
        yolo.Track(persist=persist)
    native_yolo.assert_not_called()


@pytest.mark.parametrize("persist", [False, True])
def test_tracking_passes_explicit_persistence_on_each_call(persist: bool) -> None:
    operator = yolo.Track(persist=persist)
    operator("first.jpg")
    operator("second.jpg")

    assert operator.model.track.call_args_list == [
        call(source="first.jpg", persist=persist),
        call(source="second.jpg", persist=persist),
    ]


@pytest.mark.parametrize("operator_class,method_name", METHODS)
def test_calls_accept_only_source_and_propagate_native_errors(
    operator_class, method_name: str
) -> None:
    operator = operator_class()
    method = operator.model if method_name == "__call__" else getattr(operator.model, method_name)
    with pytest.raises(TypeError):
        operator("frame.jpg", conf=0.4)
    with pytest.raises(TypeError):
        operator()
    method.assert_not_called()

    error = RuntimeError("native inference failed")
    method.side_effect = error
    with pytest.raises(RuntimeError) as raised:
        operator("frame.jpg")
    assert raised.value is error


class _PathSequence(Sequence[str]):
    def __getitem__(self, index: int) -> str:
        return ("one.jpg", "two.jpg")[index]

    def __len__(self) -> int:
        return 2


@pytest.mark.parametrize("operator_class,method_name", METHODS)
def test_generic_sequences_are_normalized_at_each_model_boundary(
    operator_class, method_name: str
) -> None:
    operator = operator_class()
    operator(_PathSequence())

    method = operator.model if method_name == "__call__" else getattr(operator.model, method_name)
    extra = {"persist": False} if operator_class is yolo.Track else {}
    assert method.call_args_list == [call(source=["one.jpg", "two.jpg"], **extra)]


@pytest.mark.parametrize(
    "source",
    [
        "frame.jpg", Path("frame.jpg"), 0, ["one.jpg"], ("one.jpg",),
        Mock(spec=Image), Mock(spec=np.ndarray), Mock(spec=Tensor),
    ],
    ids=["str", "path", "camera", "list", "tuple", "pil", "numpy", "tensor"],
)
def test_native_source_types_are_passed_through_without_copying(source) -> None:
    assert yolo._native_source(source) is source


def test_operator_description_includes_the_configured_model() -> None:
    operator = yolo.Predict(model="custom.pt", conf=0.25)
    description = repr(Pipeline([operator], auto_validate=False))
    assert "Predict(model='custom.pt', conf=0.25)" in description


def test_yolo_operator_contract_accepts_a_gathered_bgr_batch() -> None:
    # Type validation only; real pipeline execution belongs to integration CI.
    pipeline = Pipeline([Scatter(), Gather(), yolo.Predict()], auto_validate=False)
    pipeline.validate(list[npt.NDArray[np.uint8]])
