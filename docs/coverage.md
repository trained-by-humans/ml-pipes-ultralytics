# Model Functionality Coverage

Model initialization is handled as-is: construct `YOLO(model, task,
verbose)` directly, then pass that instance as the `Model` input to any
operator. The operators wrap individual model functions; they do not replace
the native model façade.

| Ultralytics `Model` functionality                                           | ml-pipes equivalent                               |
|-----------------------------------------------------------------------------|---------------------------------------------------|
| `model(source, **kwargs)` / <br>`predict(source, predictor=None, **kwargs)` | `YOLOPredict(model, **predict_options)`*          |
| `embed(source, **kwargs)`                                                   | `YOLOEmbed(model, **embed_options)`*              |
| `track(source, persist=..., **kwargs)`                                      | `YOLOTrack(model, persist=..., **track_options)`* |
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
> `ultralytics.engine.model.Model` Noneinstance (for example, `YOLO`), pass that
> same instance as the `model` constructor parameter for any `YOLO*` operator,
> and call state,
> introspection, and callback methods directly on the native model—not on the
> operator:
>
> ```python
> from ultralytics import YOLO
> from ml_pipes.ultralytics import YOLOPredict
>
> # Initialize model directly
> model = YOLO("yolo26n.pt")
> operator = YOLOPredict(model, conf=0.25)
>
> # or simply access it through operator
> operator = YOLOPredict("yolo26n.pt", conf=0.25)
> model = operator.model
>
> # call your desired function
> summary = model.info(verbose=False)
> ```
