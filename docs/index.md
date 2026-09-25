# ml-pipes-ultralytics

`ml-pipes-ultralytics` provides [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
prediction, embedding, tracking, and result operations as composable
[ml-pipes](https://github.com/trained-by-humans/ml-pipes) operators. Native
Ultralytics models and `Results` stay intact, while inference configuration and
pipeline data flow are explicit.

Install from PyPI in a Python 3.10+ environment:

```bash
python -m pip install ml-pipes-ultralytics
```

The package installs `ml-pipes-core`, `ml-pipes-vision`, and Ultralytics. See
the [Reference](reference.md) for the public operator surface and
[Coverage](coverage.md) for the native Ultralytics API comparison.

Start with the runnable [examples](https://github.com/trained-by-humans/ml-pipes-ultralytics/tree/main/examples):

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

## Licensing

`ml-pipes-ultralytics` is licensed under the [Apache License 2.0](../LICENSE.txt).
It requires Ultralytics, which is licensed separately under AGPL-3.0 or an
[Ultralytics Enterprise License](https://www.ultralytics.com/license).
Installing or using this package does not grant rights to Ultralytics software
or model weights. Community Ultralytics users must comply with AGPL-3.0;
Enterprise users must ensure their Ultralytics agreement covers their intended
use.

## Built with Ultralytics x ml-pipes

| Example | Upstream source | Note |
|---|---|---|
| [`run_detect_and_crop.py`](tutorials/detect_and_crop.md) | [Object Cropping](https://docs.ultralytics.com/guides/object-cropping/) | Detects objects, saves native result crops, and renders the annotated image. |
| [`run_segment_and_annotate.py`](tutorials/segment_and_annotate.md) | [Instance Segmentation and Tracking](https://docs.ultralytics.com/guides/instance-segmentation-and-tracking/) | Runs segmentation and renders instance-coloured masks. |
| [`run_object_blurrer.py`](tutorials/object_blurrer.md) | [Object Blurring](https://docs.ultralytics.com/guides/object-blurring/) | Tracks and blurs COCO `person` detections. |
| [`run_detection_video.py`](tutorials/detection_video.md) | [Ultralytics predict mode](https://docs.ultralytics.com/modes/predict/) | Uses an explicit OpenCV capture loop and one non-streaming prediction pipeline call per frame. |
| [`run_track_objects.py`](tutorials/track_objects.md) | [Ultralytics track mode](https://docs.ultralytics.com/modes/track/) | Preserves native tracker state and draws native tracking IDs plus explicit motion traces. |
