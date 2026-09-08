# ml-pipes-ultralytics Index

This page catalogs the public `ml_pipes.ultralytics` package surface. For the
Ultralytics `Model` API comparison, see [coverage.md](./coverage.md). For the
package overview and examples, see the repository [README.md](../README.md).

## Package Primitives

| Surface | Notes |
|---|---|
| `Source` / `Sources` | Native Ultralytics inference-source types, including paths, URLs, camera ids, PIL images, NumPy arrays, Torch tensors, and supported collections. |

## Model Inference

| Operator | Input -> Output | Notes |
|---|---|---|
| `YOLOPredict(model="yolo26n.pt", task=None, verbose=False, **predict_options)` | `Sources` -> `list[Results]` | Runs non-streaming native YOLO prediction. |
| `YOLOEmbed(model="yolo26n.pt", task=None, verbose=False, **embed_options)` | `Sources` -> `list[Tensor]` | Runs non-streaming native YOLO embedding. With no `embed` option, Ultralytics selects its default embedding layer. |
| `YOLOTrack(model="yolo26n.pt", task=None, verbose=False, persist=False, **track_options)` | `Sources` -> `list[Results]` | Runs non-streaming native YOLO tracking. `persist` controls tracker state across calls to the same operator. |

> [!IMPORTANT]
> `stream=True` is deliberately unsupported at this boundary. For videos or
> large datasets, use a pipeline specifically designed to decode, batch, and
> schedule frames before invoking these operators.

## Result Operations

`result` is the module containing the result operators below.

| Operator | Input -> Output | Notes |
|---|---|---|
| `result.To(...)` | `Results` -> `Results` | Moves native result tensors to a device or dtype. |
| `result.Plot(...)` | `Results` -> `ndarray \| Image` | Renders an annotated image. |
| `result.Show(...)` | `Results` -> `Results` | Displays an annotated image and passes the result through. |
| `result.ToDataFrame(...)` | `Results` -> `DataFrame` | Converts a result to a pandas dataframe. |
| `result.ToCSV(...)` | `Results` -> `str` | Converts a result to CSV. |
| `result.ToJSON(...)` | `Results` -> `str` | Converts a result to JSON. |
| `result.SaveTXT(...)` | `Results` -> `Results` | Writes labels and passes the result through. |
| `result.SaveCrop(...)` | `Results` -> `Results` | Writes detected-object crops and passes the result through. |
| `result.Save(...)` | `Results` -> `Results` | Saves an annotated image and passes the result through. |
| `result.Summary(...)` | `Results` -> `list[dict[str, Any]]` | Converts a result to summary dictionaries. |

The argument-less `Results` methods `cpu`, `numpy`, `cuda`, `new`, and
`verbose` can be used directly as unbound methods in a pipeline.
