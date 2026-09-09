# ml-pipes-ultralytics

[Ultralytics](https://github.com/ultralytics/ultralytics) is the open-source
repository behind the YOLO computer-vision framework. It provides model
training, validation, prediction, tracking, export, and native `Results`
objects for detection, segmentation, pose, classification, and oriented-box
tasks.

`ml-pipes-ultralytics` makes its prediction, embedding, tracking, and result
operations composable [ml-pipes](https://github.com/trained-by-humans/ml-pipes)
operators. Native Ultralytics models and `Results` remain intact at the
boundary, while pipeline configuration and data flow become explicit.

## Coverage

The package operator catalog is maintained in
[docs/INDEX.md](./docs/INDEX.md). The Ultralytics API comparison is in
[docs/coverage.md](./docs/coverage.md).

## Install

```bash
python -m pip install "ml-pipes-ultralytics @ git+https://github.com/requiem4machines/ml-pipes-ultralytics.git"
```

Ultralytics is distributed under AGPL-3.0 or an enterprise license; make sure
your use complies with its licensing terms.

## Quickstart

Build a segmentation-and-annotation pipeline. The model configuration belongs
to `yolo.Predict`; the pipeline call receives only the BGR image.

```python
import cv2

from ml_pipes.core import Pipeline
from ml_pipes.standard import Select
from ml_pipes.ultralytics import results, yolo

pipeline = Pipeline(
    [
        yolo.Predict(model="yolo26n-seg.pt", conf=0.25),
        Select(0),
        results.Plot(color_mode="instance"),
    ]
)

image = cv2.imread("photo.jpg")
annotated = pipeline(image)
cv2.imwrite("annotated.jpg", annotated)
```

`yolo.Predict`, `yolo.Embed`, and `yolo.Track` accept normal Ultralytics
source types—paths, URLs, camera sources, in-memory images, tensors, and
batches—but do not support `stream=True`. For videos or large datasets, use a
pipeline that explicitly owns decoding, batching, and frame scheduling.

## Built with Ultralytics x ml-pipes

<details>
<summary>View runnable Ultralytics example pipelines</summary>

| Example | Upstream source | Note |
|---|---|---|
| [`run_detect_and_crop.py`](./examples/run_detect_and_crop.py) | [Object Cropping](https://docs.ultralytics.com/guides/object-cropping/) | Detects objects, saves native result crops, and renders the annotated image. |
| [`run_segment_and_annotate.py`](./examples/run_segment_and_annotate.py) | [Instance Segmentation and Tracking](https://docs.ultralytics.com/guides/instance-segmentation-and-tracking/) | Runs segmentation and renders instance-coloured masks. |
| [`run_object_blurrer.py`](./examples/run_object_blurrer.py) | [Object Blurring](https://docs.ultralytics.com/guides/object-blurring/) | Tracks and blurs COCO `person` detections. |
| [`run_detection_video.py`](./examples/run_detection_video.py) | [Ultralytics predict mode](https://docs.ultralytics.com/modes/predict/) | Uses an explicit OpenCV capture loop and one non-streaming prediction pipeline call per frame. |
| [`run_track_objects.py`](./examples/run_track_objects.py) | [Ultralytics track mode](https://docs.ultralytics.com/modes/track/) | Preserves native tracker state and draws native tracking IDs plus explicit motion traces. |

</details>
