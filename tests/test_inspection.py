from unittest.mock import Mock

import pytest
from ultralytics.engine.results import Results

import ml_pipes.ultralytics  # Register the package's inspection formatters.
from ml_pipes.inspection import PipelineInspector, TextBlock

EMPTY_ROWS = {
    "path": "frame.jpg",
    "original shape": "(480, 640)",
    "detections": "0",
    "masks": "0",
    "keypoints": "0",
    "oriented boxes": "0",
    "classification": "no",
    "semantic mask": "no",
    "depth map": "no",
    "summary": "no predictions",
}


class MockResults(Results):
    """Keep native formatter dispatch, without constructing images or tensors."""

    def __init__(self, **predictions) -> None:
        self.path = "frame.jpg"
        self.orig_shape = (480, 640)
        for field in ("boxes", "masks", "probs", "keypoints", "obb", "semantic_mask", "depth"):
            setattr(self, field, predictions.get(field))
        self.verbose = Mock(return_value="  native prediction summary  \n")


def test_empty_results_inspection_is_registered_and_does_not_request_a_summary() -> None:
    result = MockResults()

    blocks = PipelineInspector()._value_to_blocks(result)

    assert len(blocks) == 1
    assert isinstance(blocks[0], TextBlock)
    assert blocks[0].title == "ultralytics.Results"
    assert dict(blocks[0].rows) == EMPTY_ROWS
    result.verbose.assert_not_called()


@pytest.mark.parametrize(
    "field,predictions,row,expected",
    [
        pytest.param("boxes", [object(), object()], "detections", "2", id="detections"),
        pytest.param("masks", [object()] * 3, "masks", "3", id="masks"),
        pytest.param("keypoints", [object()], "keypoints", "1", id="keypoints"),
        pytest.param("obb", [object()] * 4, "oriented boxes", "4", id="oriented-boxes"),
        pytest.param("probs", object(), "classification", "yes", id="classification"),
        pytest.param("semantic_mask", object(), "semantic mask", "yes", id="semantic"),
        pytest.param("depth", object(), "depth map", "yes", id="depth"),
    ],
)
def test_results_inspection_reports_each_prediction_kind_and_trims_native_summary(
    field: str, predictions, row: str, expected: str
) -> None:
    result = MockResults(**{field: predictions})

    blocks = PipelineInspector()._value_to_blocks(result)

    assert len(blocks) == 1
    assert isinstance(blocks[0], TextBlock)
    assert blocks[0].title == "ultralytics.Results"
    assert dict(blocks[0].rows) == {
        **EMPTY_ROWS, row: expected, "summary": "native prediction summary",
    }
    result.verbose.assert_called_once_with()


def test_results_inspection_tolerates_missing_optional_native_fields() -> None:
    result = MockResults(boxes=[object()])
    del result.semantic_mask
    del result.depth

    blocks = PipelineInspector()._value_to_blocks(result)

    assert dict(blocks[0].rows)["semantic mask"] == "no"
    assert dict(blocks[0].rows)["depth map"] == "no"
    result.verbose.assert_called_once_with()
