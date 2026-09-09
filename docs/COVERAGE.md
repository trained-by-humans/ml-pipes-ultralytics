# Model Functionality Coverage

Model initialization is handled as-is: construct `YOLO(model, task,
verbose)` directly, then pass that instance as the `Model` input to any
operator. The operators wrap individual model functions; they do not replace
the native model façade.

Upstream reference: [Ultralytics `Model`](https://docs.ultralytics.com/reference/engine/model/).

| Ultralytics `Model` functionality                                           | ml-pipes equivalent                               |
|-----------------------------------------------------------------------------|---------------------------------------------------|
| `model(source, **kwargs)` / <br>`predict(source, predictor=None, **kwargs)` | `yolo.Predict(model, **predict_options)`*         |
| `embed(source, **kwargs)`                                                   | `yolo.Embed(model, **embed_options)`*             |
| `track(source, persist=..., **kwargs)`                                      | `yolo.Track(model, persist=..., **track_options)`* |
| `val(validator=None, **kwargs)`                                             | None                                              |
| `train(trainer=None, **kwargs)`                                             | None                                              |
| `benchmark(**kwargs)`                                                       | None                                              |
| `export(**kwargs)`                                                          | None                                              |
| `tune(use_ray=False, iterations=10, *args, **kwargs)`                       | None                                              |
| `reset_weights()`                                                           | AS-IS                                             |
| `load(weights)`                                                             | AS-IS                                             |
| `save(filename)`                                                            | AS-IS                                             |
| `info(detailed=False, verbose=True)`                                        | AS-IS                                             |
| `fuse()`                                                                    | AS-IS                                             |
| `add_callback(event, func)`                                                 | AS-IS                                             |
| `clear_callback(event)`                                                     | AS-IS                                             |
| `reset_callbacks()`                                                         | AS-IS                                             |
| `names`                                                                     | AS-IS                                             |
| `device`                                                                    | AS-IS                                             |
| `transforms`                                                                | AS-IS                                             |
| `task_map`                                                                  | AS-IS                                             |
| `is_hub_model(model)`                                                       | AS-IS                                             |
| `is_triton_model(model)`                                                    | AS-IS                                             |

> [!IMPORTANT]
> * `stream=True` is deliberately unsupported at this boundary. For videos or
> large datasets, use a pipeline specifically designed to decode, batch, and
> schedule frames before invoking these operators.

> [!TIP]
> **AS-IS:** To use native model functionality, create or retain an
> `ultralytics.engine.model.Model` instance (for example, `YOLO`), pass that
> same instance as the `model` constructor parameter for any `yolo.*` operator,
> and call state,
> introspection, and callback methods directly on the native model—not on the
> operator:
>
> ```python
> from ultralytics import YOLO
> from ml_pipes.ultralytics import yolo
>
> # Initialize model directly
> model = YOLO("yolo26n.pt")
> operator = yolo.Predict(model, conf=0.25)
>
> # or simply access it through operator
> operator = yolo.Predict("yolo26n.pt", conf=0.25)
> model = operator.model
>
> # call your desired function
> summary = model.info(verbose=False)
> ```

## Result Functionality Coverage

`Results` fields—`orig_img`, `orig_shape`, `boxes`, `masks`, `probs`,
`keypoints`, `obb`, `semantic_mask`, `depth`, `speed`, `names`, `path`, and
`save_dir`—are pure native data. They remain accessible in a pipeline through
any operator that accepts `Results`, such as `Map`; no dedicated adapter is
needed.

Upstream reference: [Ultralytics `Results`](https://docs.ultralytics.com/reference/engine/results/).

| Ultralytics `Results` functionality        | ml-pipes operator / access                                                                            |
|--------------------------------------------|-------------------------------------------------------------------------------------------------------|
| `cpu()`                                    | Direct call: `Results.cpu`                                                                            |
| `numpy()`                                  | Direct call: `Results.numpy`                                                                          |
| `cuda()`                                   | Direct call: `Results.cuda`                                                                           |
| `new()`                                    | Direct call: `Results.new`                                                                            |
| `verbose()`                                | Direct call: `Results.verbose`                                                                        |
| `update(...)`                              | None                                                                                                  |
| `to(...)`                                  | `results.To(...)`                                                                                     |
| `plot(...)`                                | `results.Plot(...)`                                                                                   |
| `show(...)`                                | `results.Show(...)`                                                                                   |
| `to_df(...)`                               | `results.ToDataFrame(...)`                                                                            |
| `to_csv(...)`                              | `results.ToCSV(...)`                                                                                  |
| `to_json(...)`                             | `results.ToJSON(...)`                                                                                 |
| `save_txt(...)`                            | `results.SaveTXT(...)`                                                                                |
| `save_crop(...)`                           | `results.SaveCrop(...)`                                                                               |
| `save(...)`                                | `results.Save(...)`                                                                                   |
| `summary(...)`                             | `results.Summary(...)`                                                                                |

`results.Show`, `results.SaveTXT`, `results.SaveCrop`, and `results.Save` are
pipeline side effects: after performing the native action, they pass the same
`Results` object to the next operator. Transformations retain their native
return values. `update(...)` remains unimplemented while its mutation
semantics are decided.

> [!TIP]
> Methods marked **Direct call** have no required arguments beyond the native
> `Results` instance, so their unbound method can be placed directly in a
> pipeline. For example, use `Results.verbose` for a `Results` → `str` step.
