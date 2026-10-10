---
title: Process Multiple Images with YOLO
description: >-
  Learn how to run YOLO batch inference in Python, process a folder of images, and load images concurrently. Compare single-image prediction, batched inference, and parallel image decoding.
---

# Process Multiple Images efficiently with YOLO

Suppose you have a folder of images and want to detect objects—people, vehicles, animals,
and more—in every image. Running YOLO on each image individually is straightforward,
but processing images in batches can reduce the number of model calls and improve throughput.
Concurrent image loading can further reduce the time spent preparing inputs.

This tutorial walks through three approaches using 100 local images:
single-image prediction, batched inference, and batch inference with concurrent image decoding.
You'll learn how to combine these steps into a pipeline while letting YOLO handle image resizing
and result geometry. The same batching approach also applies to segmentation and embedding extraction,
with different outputs.

Before you begin, follow the [Installation](../installation.md) instructions.
Run the Python snippets in order in the same script or notebook.
Creating the prediction operator may download its model checkpoint.

## Prepare a local image archive

**Skip the download if you already have images locally.** Set `image_dir` in
the next section to your own directory instead.

For a small public archive, use [COCO128](https://docs.ultralytics.com/datasets/detect/coco128/),
Ultralytics' approximately 7 MB subset of COCO train2017. Download that archive
and copy 100 distinct JPEGs into a local directory. The other images and
training labels are not extracted; this is an inference example, not a
training dataset setup.

Run from the repository root:

```python
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

destination = Path("examples/.example_assets/coco-sample")
destination.mkdir(parents=True, exist_ok=True)
url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"

with urlopen(url, timeout=60) as response:
    archive_bytes = response.read()

with ZipFile(BytesIO(archive_bytes)) as archive:
    images = sorted(
        name
        for name in archive.namelist()
        if "/images/train2017/" in name and name.lower().endswith(".jpg")
    )[:100]
    for name in images:
        output = destination / Path(name).name
        output.write_bytes(archive.read(name))

print(f"Sample images are ready in {destination}")
```

This URL is the download declared in Ultralytics'
[COCO128 dataset configuration](https://github.com/ultralytics/ultralytics/blob/v8.4.174/ultralytics/cfg/datasets/coco128.yaml).
The images retain their upstream licenses; see [COCO's terms of use](https://cocodataset.org/#termsofuse)
before redistributing them. This download is outside all inference measurements.

From here on, every iteration starts with images already in a local directory:

```python
from pathlib import Path

image_dir = Path("examples/.example_assets/coco-sample")
sources = sorted(image_dir.glob("*.jpg"))

print(f"Processing {len(sources)} local images")
```

For the supplied sample, that is 100 JPEG files. Change the glob to match
your own archive's image format. There is no directory watcher,
export system, or manifest: the archive is just a small list of local paths.

## Process images one by one

Start with the prediction portion of the [Detect and Crop](detect_and_crop.md)
pipeline: `yolo.Predict` followed by `Select(0)`. Cropping and rendering are
optional downstream steps; this archive task only needs the object labels,
so we inspect the native result rather than writing crops.

Build the pipeline once, outside the loop:

```python
from ml_pipes.core import Pipeline
from ml_pipes.standard import Select
from ml_pipes.ultralytics import yolo

pipeline = Pipeline(
    [
        yolo.Predict(model="yolo26n.pt", conf=0.25, imgsz=640),
        Select(0),
    ],
    auto_validate=True,
)

for source in sources:
    result = pipeline(str(source))
    print(f"{source.name}: {result.verbose().strip()}")
```

Each call loads one file and predicts on one image. For 100 files, the
pipeline makes 100 model calls. An image with no detections still has a
result; `verbose()` produces a readable summary rather than a search index.

## Process several images at once

Ultralytics [documents batched prediction on an image list](https://docs.ultralytics.com/modes/predict/#key-features-of-predict-mode).
Our adapter preserves that API: pass a list to one call and receive a list
of native results, in the same order. Remove `Select(0)` so the remaining
results are not discarded.

```python
batch_pipeline = Pipeline(
    [
        yolo.Predict(model="yolo26n.pt", conf=0.25, imgsz=640)
    ],
    auto_validate=True,
)
batch_size = 8

for start in range(0, len(sources), batch_size):
    paths = sources[start : start + batch_size]
    predictions = batch_pipeline([str(path) for path in paths])
    for path, result in zip(paths, predictions):
        print(f"{path.name}: {result.verbose().strip()}")
```

The 100-image archive now needs 13 calls: twelve batches of eight and a final
batch of four. The pipeline reuses its model across batches. The length of
each input list determines this batch; `batch=` is not a scheduler that splits
an arbitrary list for us.

### Batching inference does not make file decoding concurrent

For path lists, `autocast_list` iterates over the sources, then
`LoadPilAndNumpy` converts the images serially. Batching inference therefore
does not make decoding concurrent. See the
[pinned loader source (8.4.174)](https://github.com/ultralytics/ultralytics/blob/v8.4.174/ultralytics/data/loaders.py)
and the related [batching discussion in issue #9838](https://github.com/ultralytics/ultralytics/issues/9838).

## Load and decode images concurrently

Move file loading and decoding into concurrent workers, then pass a list of
original-size BGR NumPy images to YOLO. YOLO still handles resizing and
normalization and retains the original images for native Results. See the
[documented NumPy input](https://docs.ultralytics.com/modes/predict/#inference-sources)
for format details.

```python
import numpy as np
import numpy.typing as npt

from ml_pipes.standard import Gather, Map, Scatter
from ml_pipes.vision import Decode, ImagePayload, LoadFile


def image_array(payload: ImagePayload) -> npt.NDArray[np.uint8]:
    return payload.array


concurrent_pipeline = Pipeline(
    [
        Scatter(max_concurrency=4),
        LoadFile(),
        Decode(),
        Map(image_array),
        Gather(),
        yolo.Predict(model="yolo26n.pt", conf=0.25, imgsz=640),
    ],
    auto_validate=True,
)

for start in range(0, len(sources), batch_size):
    paths = sources[start : start + batch_size]
    predictions = concurrent_pipeline([str(path) for path in paths])
    for path, result in zip(paths, predictions):
        print(f"{path.name}: {result.verbose().strip()}")
```

`Gather` preserves input order and collects images of any original dimensions.
Keep `yolo.Predict` after it: one image list produces one batched model call.
YOLO handles result geometry for the original images, so plotting or cropping
does not require reversing our own resize. Continue pairing results with
`paths`, since in-memory images receive generated result filenames.

Prepared tensors are also [supported](https://docs.ultralytics.com/modes/predict/#inference-sources),
but require caller-owned preprocessing and geometry. This guide leaves that
work to YOLO.

> [!TIP]
> `--workers` controls concurrent image loading and decoding in the runnable
> example below (`Scatter(max_concurrency=...)` in the pipeline). Adjust it
> when running multiple inference instances to avoid oversubscribing CPU resources.

## Apply the same input flow to other YOLO operations

Reuse `image_array` and the same loading steps. Only the highlighted
task-specific operator changes.

=== "Detection"

    ```python hl_lines="8"
    pipeline = Pipeline(
        [
            Scatter(max_concurrency=4),
            LoadFile(),
            Decode(),
            Map(image_array),
            Gather(),
            yolo.Predict(model="yolo26n.pt", imgsz=640, conf=0.25),
        ],
        auto_validate=True,
    )
    ```

=== "Segmentation"

    ```python hl_lines="8"
    pipeline = Pipeline(
        [
            Scatter(max_concurrency=4),
            LoadFile(),
            Decode(),
            Map(image_array),
            Gather(),
            yolo.Predict(model="yolo26n-seg.pt", imgsz=640, conf=0.25),
        ],
        auto_validate=True,
    )
    ```

=== "Depth"

    ```python hl_lines="8"
    pipeline = Pipeline(
        [
            Scatter(max_concurrency=4),
            LoadFile(),
            Decode(),
            Map(image_array),
            Gather(),
            yolo.Predict(model="yolo26n-depth.pt", imgsz=640),
        ],
        auto_validate=True,
    )
    ```

=== "Pose"

    ```python hl_lines="8"
    pipeline = Pipeline(
        [
            Scatter(max_concurrency=4),
            LoadFile(),
            Decode(),
            Map(image_array),
            Gather(),
            yolo.Predict(model="yolo26n-pose.pt", imgsz=640, conf=0.25),
        ],
        auto_validate=True,
    )
    ```

=== "Embeddings"

    ```python hl_lines="8"
    pipeline = Pipeline(
        [
            Scatter(max_concurrency=4),
            LoadFile(),
            Decode(),
            Map(image_array),
            Gather(),
            yolo.Embed(model="yolo26n.pt", imgsz=640),
        ],
        auto_validate=True,
    )
    ```

Prediction tasks return native `Results`; embeddings return one native
PyTorch tensor per image. Embedding extraction requires a native PyTorch
model, not an exported backend.

YOLO owns task-specific preprocessing; the loading pipeline does not resize
images or impose a shared tensor shape.

The end result is still the original archive task: one output associated with
each local image. Batching and concurrent preparation are ways to finish that
task sooner, not a separate file-processing system.

## Run the example

Run from the repository root. With no `--source`, the example downloads the
same 100 COCO128 images prepared above on its first run, then reuses them.
It prints each image's prediction summary; it does not save or modify images.

```bash
python examples/run_batch_predict.py
```

Run each tutorial stage on your own image directory:

```bash
python examples/run_batch_predict.py --source path/to/images --mode single
python examples/run_batch_predict.py --source path/to/images --mode batch --batch-size 8
python examples/run_batch_predict.py --source path/to/images --mode concurrent --batch-size 8 --workers 4
```

The default mode is `concurrent`. Use `--pattern '*.png'` for PNG images,
`--model` for another prediction model, and `--device` to select a device.
YOLO handles resizing and result geometry in every mode. These commands
demonstrate the input flows, not a benchmark.

See [`run_batch_predict.py`](https://github.com/trained-by-humans/ml-pipes-ultralytics/blob/main/examples/run_batch_predict.py)
for all command-line options.
