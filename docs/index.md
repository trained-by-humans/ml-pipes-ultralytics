# ml-pipes-ultralytics

`ml-pipes-ultralytics` provides [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
prediction, embedding, tracking, and result operations as composable
[ml-pipes](https://github.com/trained-by-humans/ml-pipes) operators. Native
Ultralytics models and `Results` stay intact, while inference configuration and
pipeline data flow are explicit.

Install directly from this repository in a Python 3.10+ environment:

```bash
python -m pip install "ml-pipes-ultralytics @ git+https://github.com/requiem4machines/ml-pipes-ultralytics.git"
```

The package installs `ml-pipes-core`, `ml-pipes-vision`, and Ultralytics. See
the [Reference](reference.md) for the public operator surface and
[Coverage](coverage.md) for the native Ultralytics API comparison.

Start with the runnable [examples](https://github.com/requiem4machines/ml-pipes-ultralytics/tree/main/examples):

```python
from ml_pipes.core import Pipeline
from ml_pipes.standard import Select
from ml_pipes.ultralytics import results, yolo

pipeline = Pipeline(
    [
        yolo.Predict("yolo26n-seg.pt", conf=0.25),
        Select(0),
        results.Plot(color_mode="instance"),
    ]
)
```

## Built with Ultralytics x ml-pipes

| Example | Upstream source | Note |
|---|---|---|
| [`run_detect_and_crop.py`](https://github.com/requiem4machines/ml-pipes-ultralytics/blob/main/examples/run_detect_and_crop.py) | [Object Cropping](https://docs.ultralytics.com/guides/object-cropping/) | Detects objects, saves native result crops, and renders the annotated image. |
| [`run_segment_and_annotate.py`](https://github.com/requiem4machines/ml-pipes-ultralytics/blob/main/examples/run_segment_and_annotate.py) | [Instance Segmentation and Tracking](https://docs.ultralytics.com/guides/instance-segmentation-and-tracking/) | Runs segmentation and renders instance-coloured masks. |
| [`run_object_blurrer.py`](https://github.com/requiem4machines/ml-pipes-ultralytics/blob/main/examples/run_object_blurrer.py) | [Object Blurring](https://docs.ultralytics.com/guides/object-blurring/) | Tracks and blurs COCO `person` detections. |
| [`run_detection_video.py`](https://github.com/requiem4machines/ml-pipes-ultralytics/blob/main/examples/run_detection_video.py) | [Ultralytics predict mode](https://docs.ultralytics.com/modes/predict/) | Uses an explicit OpenCV capture loop and one non-streaming prediction pipeline call per frame. |
| [`run_track_objects.py`](https://github.com/requiem4machines/ml-pipes-ultralytics/blob/main/examples/run_track_objects.py) | [Ultralytics track mode](https://docs.ultralytics.com/modes/track/) | Preserves native tracker state and draws native tracking IDs plus explicit motion traces. |
