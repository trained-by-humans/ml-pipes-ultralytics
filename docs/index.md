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
