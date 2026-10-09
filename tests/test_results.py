from __future__ import annotations

from pathlib import Path
from typing import get_args, get_type_hints
from unittest.mock import Mock, create_autospec, sentinel

import numpy as np
import numpy.typing as npt
import pytest
from PIL.Image import Image
from polars import DataFrame
from torch import float32
from ultralytics.engine.results import Results

from ml_pipes.ultralytics import results as result_ops

CONVERSIONS = [
    (result_ops.ToDataFrame, "to_df"),
    (result_ops.ToCSV, "to_csv"),
    (result_ops.ToJSON, "to_json"),
    (result_ops.Summary, "summary"),
]
SIDE_EFFECTS = [
    pytest.param(result_ops.Show, "show", (False,), {"line_width": 2}, id="show"),
    pytest.param(
        result_ops.SaveTXT, "save_txt", ("labels.txt",), {"save_conf": True}, id="save-txt"
    ),
    pytest.param(result_ops.SaveCrop, "save_crop", ("crops", "frame"), {}, id="save-crop"),
    pytest.param(
        result_ops.Save, "save", ("image.jpg", False), {"line_width": 2}, id="save"
    ),
]
PLOT_DEFAULTS = {
    "conf": True,
    "line_width": None,
    "font_size": None,
    "font": "Arial.ttf",
    "pil": False,
    "img": None,
    "kpt_radius": 5,
    "kpt_line": True,
    "labels": True,
    "boxes": True,
    "masks": True,
    "probs": True,
    "show": False,
    "save": False,
    "filename": None,
    "color_mode": "class",
    "txt_color": (255, 255, 255),
}
PLOT_OPTIONS = {
    "conf": False,
    "line_width": 3,
    "font_size": 12,
    "font": "custom.ttf",
    "pil": True,
    "img": sentinel.image,
    "kpt_radius": 7,
    "kpt_line": False,
    "labels": False,
    "boxes": False,
    "masks": False,
    "probs": False,
    "show": True,
    "save": True,
    "filename": "annotated.jpg",
    "color_mode": "instance",
    "txt_color": (1, 2, 3),
}


@pytest.fixture
def native_result() -> Mock:
    # All methods are checked against the installed Ultralytics signatures.
    return create_autospec(Results, instance=True, spec_set=True)


def test_to_forwards_device_and_dtype_and_returns_the_transformed_result(native_result: Mock) -> None:
    transformed = create_autospec(Results, instance=True, spec_set=True)
    native_result.to.return_value = transformed

    assert result_ops.To("cpu", dtype=float32, non_blocking=True)(native_result) is transformed
    assert transformed is not native_result
    native_result.to.assert_called_once_with("cpu", dtype=float32, non_blocking=True)


@pytest.mark.parametrize("options", [{}, PLOT_OPTIONS], ids=["defaults", "configured"])
def test_plot_forwards_every_option_and_preserves_numpy_or_pil_output(
    options: dict, native_result: Mock
) -> None:
    output = Mock(spec=Image if options.get("pil", False) else np.ndarray)
    native_result.plot.return_value = output

    assert result_ops.Plot(**options)(native_result) is output
    native_result.plot.assert_called_once_with(**(PLOT_DEFAULTS | options))


@pytest.mark.parametrize("operator_class,method_name", CONVERSIONS)
@pytest.mark.parametrize(
    "options", [{}, {"normalize": True, "decimals": 3}], ids=["defaults", "configured"]
)
def test_conversions_forward_format_options_and_preserve_native_output(
    operator_class, method_name: str, options: dict, native_result: Mock
) -> None:
    method = getattr(native_result, method_name)
    method.return_value = sentinel.native_output

    assert operator_class(**options)(native_result) is sentinel.native_output
    method.assert_called_once_with(**({"normalize": False, "decimals": 5} | options))


@pytest.mark.parametrize("operator_class,method_name,args,kwargs", SIDE_EFFECTS)
def test_side_effects_forward_configuration_but_return_the_original_result(
    operator_class, method_name: str, args: tuple, kwargs: dict, native_result: Mock
) -> None:
    method = getattr(native_result, method_name)
    method.return_value = sentinel.ignored_native_return

    assert operator_class(*args, **kwargs)(native_result) is native_result
    method.assert_called_once_with(*args, **kwargs)


@pytest.mark.parametrize(
    "operator,method_name,args,kwargs",
    [
        pytest.param(result_ops.Show(), "show", (), {}, id="show"),
        pytest.param(
            result_ops.SaveTXT("labels.txt"), "save_txt", ("labels.txt",),
            {"save_conf": False}, id="save-txt",
        ),
        pytest.param(
            result_ops.SaveCrop("crops"), "save_crop", ("crops", Path("im.jpg")),
            {}, id="save-crop",
        ),
        pytest.param(result_ops.Save(), "save", (None,), {}, id="save"),
    ],
)
def test_side_effect_default_configuration(
    operator, method_name: str, args: tuple, kwargs: dict, native_result: Mock
) -> None:
    assert operator(native_result) is native_result
    getattr(native_result, method_name).assert_called_once_with(*args, **kwargs)


@pytest.mark.parametrize("operator_class,method_name,args,kwargs", SIDE_EFFECTS)
def test_side_effect_configuration_can_be_reused_for_another_result(
    operator_class, method_name: str, args: tuple, kwargs: dict, native_result: Mock
) -> None:
    other_result = create_autospec(Results, instance=True, spec_set=True)
    operator = operator_class(*args, **kwargs)

    assert operator(native_result) is native_result
    assert operator(other_result) is other_result
    getattr(native_result, method_name).assert_called_once_with(*args, **kwargs)
    getattr(other_result, method_name).assert_called_once_with(*args, **kwargs)


@pytest.mark.parametrize(
    "operator,method_name",
    [
        pytest.param(result_ops.To(), "to", id="to"),
        pytest.param(result_ops.Plot(), "plot", id="plot"),
        *[pytest.param(operator_class(), name, id=name) for operator_class, name in CONVERSIONS],
        pytest.param(result_ops.Show(), "show", id="show"),
        pytest.param(result_ops.SaveTXT("labels.txt"), "save_txt", id="save-txt"),
        pytest.param(result_ops.SaveCrop("crops"), "save_crop", id="save-crop"),
        pytest.param(result_ops.Save(), "save", id="save"),
    ],
)
def test_native_result_errors_are_not_swallowed(
    operator, method_name: str, native_result: Mock
) -> None:
    error = RuntimeError("native result operation failed")
    method = getattr(native_result, method_name)
    method.side_effect = error

    with pytest.raises(RuntimeError) as raised:
        operator(native_result)
    assert raised.value is error
    method.assert_called_once()


def test_return_annotations_describe_native_dataframe_and_plot_outputs() -> None:
    assert get_type_hints(result_ops.ToDataFrame.__call__)["return"] is DataFrame
    plot_types = get_args(get_type_hints(result_ops.Plot.__call__)["return"])
    assert set(plot_types) == {npt.NDArray[np.uint8], Image}
