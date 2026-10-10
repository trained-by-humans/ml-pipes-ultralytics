<p align="center">
  <img src="assets/stacked-title.svg" alt="ml-pipes-ultralytics stacked-text logo" width="640">
</p>

<p align="center">
  <a href="https://pypi.org/project/ml-pipes-ultralytics/"><img src="https://img.shields.io/pypi/v/ml-pipes-ultralytics?style=flat-square&amp;logo=pypi&amp;logoColor=white&amp;label=PyPI&amp;color=C48800" alt="Latest PyPI release"></a>
  <a href="https://pypi.org/project/ml-pipes-ultralytics/"><img src="https://img.shields.io/badge/Python-3.10%2B-2377C8?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python 3.10 or newer"></a>
  <a href="https://github.com/trained-by-humans/ml-pipes-ultralytics/blob/main/LICENSE.txt"><img src="https://img.shields.io/pypi/l/ml-pipes-ultralytics?style=flat-square&amp;label=License&amp;color=287C35" alt="Apache 2.0 license"></a>
  <a href="docs/coverage.md"><img src="https://img.shields.io/badge/Ultralytics-8.4.143-0B23A9?style=flat-square" alt="Ultralytics 8.4.143 API coverage"></a>
</p>

<p align="center">
  <a href="https://github.com/ultralytics/ultralytics">Ultralytics YOLO</a> capabilities as composable operators in <a href="https://github.com/trained-by-humans/ml-pipes">ml-pipes</a>.<br>
  Build explicit computer-vision pipelines for prediction, embedding, tracking, and result processing.
</p>

> [!IMPORTANT] 
> `ml-pipes-ultralytics` remains a community-maintained operator package
> within the [ml-pipes](https://github.com/trained-by-humans) ecosystem. The
> project has its own maintainers and development roadmap, while benefiting
> from ml-pipes' verified publishing and distribution process. 
> 
> For contributions, issues, and project decisions, use this repository's
> maintainers and issue tracker.

## Coverage

The package operator catalog is maintained in
[docs/reference.md](./docs/reference.md). The Ultralytics API comparison is in
[docs/coverage.md](./docs/coverage.md).

## Install

```bash
python -m pip install ml-pipes-ultralytics
```

## License and Ultralytics terms

`ml-pipes-ultralytics` is licensed under the [Apache License 2.0](./LICENSE.txt).
It requires [Ultralytics](https://github.com/ultralytics/ultralytics), which is
licensed separately under AGPL-3.0 or an [Ultralytics Enterprise
License](https://www.ultralytics.com/license). Installing or using this package
does not grant rights to Ultralytics software or model weights. Users of the
community Ultralytics distribution must comply with AGPL-3.0; Enterprise users
must ensure their Ultralytics agreement covers their intended use.

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
| [`run_batch_predict.py`](./examples/run_batch_predict.py) | [Ultralytics predict mode](https://docs.ultralytics.com/modes/predict/) | Processes a local image archive with single-image prediction, batching, or batched inference with concurrent image decoding. |

</details>

For additional YOLO examples and broader computer-vision use cases, see
[ml-pipes-supervision](https://github.com/trained-by-humans/ml-pipes-supervision).
