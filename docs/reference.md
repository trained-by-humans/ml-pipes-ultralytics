# ml-pipes-ultralytics Index

This page catalogs the public `ml_pipes.ultralytics` package surface.

For the Ultralytics `Model` API comparison, see [coverage.md](coverage.md).
For the package overview and examples, see the repository [README](https://github.com/requiem4machines/ml-pipes-ultralytics#readme).

## Public Modules

| Module | Contents |
|---|---|
| `yolo` | `Predict`, `Embed`, and `Track` operators, plus the native source and configuration type aliases. |
| `results` | Operators wrapping configurable native `Results` methods. |

## Model Inference

See the upstream [Ultralytics `Model` reference](https://docs.ultralytics.com/reference/engine/model/).

| Operator | Input -> Output | Notes |
|---|---|---|
| `yolo.Predict(model="yolo26n.pt", task=None, verbose=False, **predict_options)` | `Sources` -> `list[Results]` | Runs non-streaming native YOLO prediction. |
| `yolo.Embed(model="yolo26n.pt", task=None, verbose=False, **embed_options)` | `Sources` -> `list[Tensor]` | Runs non-streaming native YOLO embedding. With no `embed` option, Ultralytics selects its default embedding layer. |
| `yolo.Track(model="yolo26n.pt", task=None, verbose=False, persist=False, **track_options)` | `Sources` -> `list[Results]` | Runs non-streaming native YOLO tracking. `persist` controls tracker state across calls to the same operator. |

> [!IMPORTANT]
> `stream=True` is deliberately unsupported at this boundary. For videos or
> large datasets, use a pipeline specifically designed to decode, batch, and
> schedule frames before invoking these operators.

## Result Operations

`results` is the module containing the result operators below.
See the upstream [Ultralytics `Results` reference](https://docs.ultralytics.com/reference/engine/results/).

| Operator | Input -> Output | Notes |
|---|---|---|
| `results.To(...)` | `Results` -> `Results` | Moves native result tensors to a device or dtype. |
| `results.Plot(...)` | `Results` -> `ndarray \| Image` | Renders an annotated image. |
| `results.Show(...)` | `Results` -> `Results` | Displays an annotated image and passes the result through. |
| `results.ToDataFrame(...)` | `Results` -> `DataFrame` | Converts a result to a pandas dataframe. |
| `results.ToCSV(...)` | `Results` -> `str` | Converts a result to CSV. |
| `results.ToJSON(...)` | `Results` -> `str` | Converts a result to JSON. |
| `results.SaveTXT(...)` | `Results` -> `Results` | Writes labels and passes the result through. |
| `results.SaveCrop(...)` | `Results` -> `Results` | Writes detected-object crops and passes the result through. |
| `results.Save(...)` | `Results` -> `Results` | Saves an annotated image and passes the result through. |
| `results.Summary(...)` | `Results` -> `list[dict[str, Any]]` | Converts a result to summary dictionaries. |

The argument-less `Results` methods `cpu`, `numpy`, `cuda`, `new`, and
`verbose` can be used directly as unbound methods in a pipeline.
