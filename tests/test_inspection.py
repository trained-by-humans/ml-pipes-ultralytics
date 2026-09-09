import numpy as np
from ultralytics.engine.results import Results

import ml_pipes.ultralytics  # Register the package's inspection formatters.
from ml_pipes.inspection import PipelineInspector, TextBlock


def test_ultralytics_results_inspection_renders_result_summary() -> None:
    bgr = np.array([[[0, 0, 255]]], dtype=np.uint8)
    result = Results(orig_img=bgr, path="frame.jpg", names={})

    blocks = PipelineInspector()._value_to_blocks(result)

    assert len(blocks) == 1
    assert isinstance(blocks[0], TextBlock)
    assert blocks[0].title == "ultralytics.Results"
    assert ("original shape", "(1, 1)") in blocks[0].rows
    assert ("detections", "0") in blocks[0].rows
