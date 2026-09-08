# ml-pipes-ultralytics Index

This page catalogs the public `ml_pipes.ultralytics` package surface. For the
Ultralytics `Model` API comparison, see [coverage.md](./coverage.md). For the
package overview and examples, see the repository [README.md](../README.md).

## Package Primitives

| Surface | Notes |
|---|---|
| `Model` | A model path, `Path`, or existing native `ultralytics.engine.model.Model`. Passing an existing model shares it between operators. |
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
