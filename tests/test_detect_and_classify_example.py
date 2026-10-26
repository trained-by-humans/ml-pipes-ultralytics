"""Check per-detection classification without downloading or running models."""
from __future__ import annotations

import ast
import importlib
import re
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, create_autospec

import cv2
import numpy as np
import pytest
from ultralytics.engine.model import Model
from ultralytics.engine.results import Results

from ml_pipes.ultralytics import results, yolo

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
DETECTOR_NAMES = {0: "dog", 1: "bear", 2: "teddy bear"}
CLASSIFIER_NAMES = {
    0: "brown_bear", 1: "teddy_bear", 2: "giant_panda", 3: "dog",
    4: "ice_bear", 5: "sloth_bear", 6: "American_black_bear", 7: "lesser_panda",
}
BEAR_CLASS_IDS = (0, 2, 4, 5, 6, 7)


@pytest.fixture(autouse=True)
def no_font_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    # Native plotting falls back to Pillow's bundled font if Arial is absent.
    # Unit tests should never attempt Ultralytics' optional font download.
    monkeypatch.setattr("ultralytics.utils.plotting.check_font", lambda font: font)


@pytest.fixture
def example(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(EXAMPLES))
    return importlib.import_module("run_detect_and_classify")


@pytest.fixture
def classified_inputs(example: ModuleType, monkeypatch: pytest.MonkeyPatch) -> list[tuple[Results, list[Results]]]:
    """Observe the classification boundary without retaining it in production."""
    captured = []
    label_detections = example.LabelDetections.__call__

    def capture(self, detections: Results, classifications: list[Results]) -> Results:
        captured.append((detections, classifications))
        return label_detections(self, detections, classifications)

    monkeypatch.setattr(example.LabelDetections, "__call__", capture)
    return captured


def detection_result(image: np.ndarray, boxes: list[list[float]]) -> Results:
    return Results(image, path="source.png", names=DETECTOR_NAMES, boxes=np.array(boxes, dtype=np.float32).reshape(-1, 6))


def classifier_model(probabilities: list[list[float]]) -> Mock:
    model = create_autospec(Model, instance=True, spec_set=True)
    scores = iter(probabilities)

    def predict(source, **options):
        return [Results(source, path="crop.jpg", names=CLASSIFIER_NAMES, probs=np.array(next(scores), dtype=np.float32))]

    model.side_effect = predict
    return model


def test_filter_matches_exact_label_and_preserves_native_data(example: ModuleType) -> None:
    image = np.zeros((12, 18, 3), dtype=np.uint8)
    result = detection_result(image, [[0, 0, 2, 2, 0.9, 0], [1, 2, 5, 6, 0.8, 1], [2, 2, 6, 6, 0.7, 2], [4, 5, 9, 10, 0.6, 1]])

    selected = example.FilterResults("bear")(result)

    assert isinstance(selected, Results) and selected.orig_img is image
    assert selected.boxes.cls.tolist() == [1, 1]
    np.testing.assert_array_equal(selected.boxes.data, result.boxes.data[[1, 3]])
    assert len(result.boxes) == 4


@pytest.mark.parametrize("boxes", [None, np.empty((0, 6), dtype=np.float32), np.array([[1, 2, 5, 6, 0.8, 1]], dtype=np.float32)])
def test_no_detections_skips_classifier(example: ModuleType, tmp_path: Path, boxes: np.ndarray | None) -> None:
    image = np.zeros((12, 18, 3), dtype=np.uint8)
    detector = create_autospec(Model, instance=True, spec_set=True)
    detector.return_value = [Results(image, path="source.png", names=DETECTOR_NAMES, boxes=boxes)]
    classifier = classifier_model([])

    pipeline = example.build_pipeline(detector, classifier, output=tmp_path / "annotated.jpg")
    pipeline.validate()

    annotated = pipeline(image)

    np.testing.assert_array_equal(annotated, image)
    np.testing.assert_array_equal(cv2.imread(str(tmp_path / "annotated.jpg")), image)
    detector.assert_called_once()
    classifier.assert_not_called()


def test_missing_class_is_an_error_even_with_no_detections(example: ModuleType) -> None:
    result = Results(np.zeros((12, 18, 3), dtype=np.uint8), path="source.png", names={0: "dog"})

    with pytest.raises(ValueError, match="Detector has no class named 'bear'"):
        example.FilterResults("bear")(result)


@pytest.mark.parametrize("tensor", [False, True])
def test_crops_use_native_rounding_without_changing_detections_or_source(example: ModuleType, tensor: bool) -> None:
    image = np.arange(12 * 18 * 3, dtype=np.uint8).reshape(12, 18, 3)
    original = image.copy()
    result = detection_result(image, [[-2.4, -1.3, 5.2, 4.1, 0.8, 1], [15.2, 8.4, 22.7, 20.3, 0.6, 1]])
    original_boxes = result.boxes.data.copy()
    if tensor:
        import torch

        result.update(boxes=torch.from_numpy(original_boxes.copy()))
        # update() clips coordinates, so restore the out-of-bounds test inputs.
        result.boxes.data[:] = torch.from_numpy(original_boxes)

    detections, crops = example.CropResults()(example.FilterResults("bear")(result))

    assert isinstance(detections, Results)
    np.testing.assert_array_equal(detections.boxes.cpu().numpy().data, original_boxes)
    assert detections.boxes.conf.tolist() == pytest.approx([0.8, 0.6])
    np.testing.assert_array_equal(result.boxes.cpu().numpy().data, original_boxes)
    np.testing.assert_array_equal(crops[0], image[:4, :5])
    np.testing.assert_array_equal(crops[1], image[8:12, 15:18])
    assert not np.shares_memory(crops[0], image)
    crops[0][:] = 0
    np.testing.assert_array_equal(image, original)


def test_crop_results_forwards_configuration_to_the_native_helper(example: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.zeros((12, 18, 3), dtype=np.uint8)
    result = detection_result(image, [[1, 2, 5, 6, 0.8, 1]])
    native_crop = np.ones((4, 4, 3), dtype=np.uint8)
    crop = Mock(return_value=native_crop)
    monkeypatch.setattr(example, "save_one_box", crop)

    detections, crops = example.CropResults(save=True, BGR=False, gain=1.5, pad=2)(result)

    crop.assert_called_once()
    np.testing.assert_array_equal(crop.call_args.args[0], result.boxes.xyxy)
    assert crop.call_args.args[1] is image
    assert crop.call_args.kwargs == {"save": True, "BGR": False, "gain": 1.5, "pad": 2}
    np.testing.assert_array_equal(detections.boxes.data, result.boxes.data)
    np.testing.assert_array_equal(crops[0], native_crop)
    assert not np.shares_memory(crops[0], native_crop)


def test_crop_results_defaults_keep_bgr_crops_in_memory(
    example: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.full((12, 18, 3), [10, 20, 200], dtype=np.uint8)
    result = detection_result(image, [[4, 4, 8, 8, 0.8, 1]])
    monkeypatch.chdir(tmp_path)

    _, crops = example.CropResults()(result)

    np.testing.assert_array_equal(crops[0], image[4:8, 4:8])
    assert list(tmp_path.iterdir()) == []


def test_crop_results_supports_rgb_and_native_box_expansion(example: ModuleType) -> None:
    image = np.arange(12 * 18 * 3, dtype=np.uint8).reshape(12, 18, 3)
    result = detection_result(image, [[4, 4, 8, 8, 0.8, 1]])

    detections, crops = example.CropResults(BGR=False, gain=1.5, pad=2)(result)

    np.testing.assert_array_equal(crops[0], image[2:10, 2:10, ::-1])
    np.testing.assert_array_equal(detections.boxes.data, result.boxes.data)


@pytest.mark.parametrize(
    "coordinates,diagnostic",
    [([18, 1, 24, 5], "empty"), ([4, 4, 4, 8], "empty"), ([8, 4, 2, 8], "empty"),
     ([float("nan"), 1, 4, 5], "non-finite"), ([0, 1, float("inf"), 5], "non-finite")],
)
def test_invalid_boxes_are_skipped_with_diagnostic(example: ModuleType, coordinates: list[float], diagnostic: str) -> None:
    result = detection_result(np.zeros((12, 18, 3), dtype=np.uint8), [[*coordinates, 0.8, 1]])

    with pytest.warns(RuntimeWarning, match=f"Skipping detection 0: {diagnostic}"):
        detections, crops = example.CropResults()(example.FilterResults("bear")(result))

    assert len(detections.boxes) == 0
    assert crops == []


def test_pipeline_classifies_each_crop_and_ranks_only_bears_and_pandas(
    example: ModuleType, tmp_path: Path, classified_inputs: list[tuple[Results, list[Results]]],
) -> None:
    image = np.arange(12 * 18 * 3, dtype=np.uint8).reshape(12, 18, 3)
    original = image.copy()
    detector = create_autospec(Model, instance=True, spec_set=True)
    detector.return_value = [detection_result(image, [[0, 0, 2, 2, 0.9, 0], [1, 2, 5, 6, 0.8, 1], [4, 5, 9, 10, 0.6, 1]])]
    classifier = classifier_model([
        [0.05, 0.6, 0.1, 0.13, 0.07, 0.03, 0.01, 0.01],
        [0.1, 0.05, 0.7, 0.03, 0.08, 0.02, 0.01, 0.01],
    ])
    pipeline = example.build_pipeline(
        detector, classifier, output=tmp_path / "annotated.jpg", class_name="bear", classification_classes=BEAR_CLASS_IDS,
        conf=0.4, iou=0.5, det_imgsz=320, cls_imgsz=128, device="cpu",
    )
    pipeline.validate()

    annotated = pipeline(image)
    assert len(classified_inputs) == 1
    detections, classifications = classified_inputs[0]
    ranked = [
        [(class_id, result.names[class_id], float(result.probs.data[class_id])) for class_id in result.probs.top5]
        for result in classifications
    ]

    assert detections.boxes.xyxy.tolist() == [[1, 2, 5, 6], [4, 5, 9, 10]]
    assert detections.boxes.cls.tolist() == [1, 1]
    assert detections.boxes.conf.tolist() == pytest.approx([0.8, 0.6])
    assert detections.names == DETECTOR_NAMES
    assert [predictions[0][1] for predictions in ranked] == ["giant_panda", "giant_panda"]
    assert [predictions[0][2] for predictions in ranked] == pytest.approx([0.1, 0.7])
    assert [class_id for class_id, _, _ in ranked[0]] == [BEAR_CLASS_IDS.index(class_id) for class_id in (2, 4, 0, 5, 6)]
    assert [confidence for _, _, confidence in ranked[0]] == pytest.approx([0.1, 0.07, 0.05, 0.03, 0.01])
    assert all(label in {CLASSIFIER_NAMES[class_id] for class_id in BEAR_CLASS_IDS} for predictions in ranked for _, label, _ in predictions)
    np.testing.assert_allclose(classifications[0].probs.data, [0.05, 0.1, 0.07, 0.03, 0.01, 0.01])
    assert np.asarray(annotated).shape == image.shape
    assert classifier.call_count == 2
    assert detector.call_args.kwargs.keys() == {"source", "conf", "iou", "imgsz", "device"}
    assert detector.call_args.kwargs["conf"] == 0.4
    assert detector.call_args.kwargs["iou"] == 0.5
    assert detector.call_args.kwargs["imgsz"] == 320
    for call, crop in zip(classifier.call_args_list, (image[2:6, 1:5], image[5:10, 4:9])):
        np.testing.assert_array_equal(call.kwargs["source"], crop)
        assert {key: value for key, value in call.kwargs.items() if key != "source"} == {"imgsz": 128, "device": "cpu"}
    np.testing.assert_array_equal(image, original)


def test_invalid_crops_do_not_shift_classification_alignment(
    example: ModuleType, tmp_path: Path, classified_inputs: list[tuple[Results, list[Results]]],
) -> None:
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    detector = create_autospec(Model, instance=True, spec_set=True)
    detector.return_value = [detection_result(image, [[10, 3, 5, 25, 0.9, 1], [20, 2, 30, 28, 0.8, 1]])]
    classifier = classifier_model([[0.6, 0.1, 0.3, 0, 0, 0, 0, 0]])

    with pytest.warns(RuntimeWarning, match="empty bounding-box crop"):
        annotated = example.build_pipeline(
            detector, classifier, output=tmp_path / "annotated.jpg", class_name="bear", classification_classes=BEAR_CLASS_IDS,
        )(image)

    assert len(classified_inputs) == 1
    detections, classifications = classified_inputs[0]
    assert detections.boxes.xyxy.tolist() == [[20, 2, 30, 28]]
    assert len(classifications) == 1
    assert classifications[0].orig_img.shape == (26, 10, 3)
    classifier.assert_called_once()
    assert np.asarray(annotated).shape == image.shape


def test_label_generation_and_native_plot_are_pipeline_steps(
    example: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    classified_inputs: list[tuple[Results, list[Results]]],
) -> None:
    from PIL import Image

    image = np.zeros((32, 48, 3), dtype=np.uint8)
    detector = create_autospec(Model, instance=True, spec_set=True)
    detector.return_value = [detection_result(image, [[1, 2, 25, 28, 0.8, 1], [26, 2, 45, 28, 0.9, 1]])]
    classifier = classifier_model([[0.4, 0, 0.3, 0.2, 0.1, 0, 0, 0], [0.7, 0, 0.2, 0, 0.1, 0, 0, 0]])
    calls = []

    def plot(result, **options):
        calls.append((result, options))
        return Image.new("RGB", (48, 32), (10, 20, 30))

    monkeypatch.setattr(Results, "plot", plot)
    annotated = example.build_pipeline(
        detector, classifier, output=tmp_path / "annotated.jpg", show=True,
        class_name="bear", classification_classes=BEAR_CLASS_IDS,
    )(image)

    assert len(calls) == 1
    labeled, options = calls[0]
    assert labeled.names == {0: "brown bear / giant panda", 1: "brown bear"}
    assert options["conf"] is False
    assert options["pil"] is True
    assert options["font_size"] == 18
    assert options["line_width"] == 2
    assert options["save"] is True
    assert options["show"] is True
    assert options["filename"] == str(tmp_path / "annotated.jpg")
    assert annotated.getpixel((0, 0)) == (10, 20, 30)  # No conversion is needed for native saving.
    assert len(classified_inputs) == 1
    detections, classifications = classified_inputs[0]
    assert detections.names == DETECTOR_NAMES
    assert detections.boxes.cls.tolist() == [1, 1]
    assert len(classifications) == 2


@pytest.mark.parametrize("tensor", [False, True])
def test_plot_labels_do_not_mutate_tracked_detection_ids_or_scores(example: ModuleType, tensor: bool) -> None:
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    boxes = np.array([[1, 2, 25, 28, 42, 0.8, 1]], dtype=np.float32)
    if tensor:
        import torch

        boxes = torch.from_numpy(boxes)
    detections = Results(image, path="source.png", names=DETECTOR_NAMES, boxes=boxes)
    classification = Results(image, path="crop.jpg", names={0: "brown_bear"}, probs=np.array([1.0]))

    labeled = example.LabelDetections()(detections, [classification])

    assert labeled.names == {0: "brown bear"}
    assert labeled.boxes.data.shape == (1, 6)
    assert not labeled.boxes.is_track
    assert labeled.boxes.conf.item() == pytest.approx(0.8)
    assert detections.boxes.is_track
    assert detections.boxes.id.item() == 42
    assert detections.boxes.cls.item() == 1
    assert detections.names == DETECTOR_NAMES


def test_label_generation_rejects_unaligned_collections(example: ModuleType) -> None:
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    detections = detection_result(image, [[1, 2, 25, 28, 0.8, 1]])

    with pytest.raises(ValueError, match="one classification result"):
        example.LabelDetections()(detections, [])


def test_native_plot_saves_without_a_color_conversion_step(example: ModuleType, tmp_path: Path) -> None:
    image = np.full((32, 48, 3), [10, 20, 200], dtype=np.uint8)
    detector = create_autospec(Model, instance=True, spec_set=True)
    detector.return_value = [detection_result(image, [])]
    classifier = classifier_model([])
    output = tmp_path / "annotated.png"

    annotated = example.build_pipeline(detector, classifier, output=output)(image)

    assert annotated.getpixel((0, 0)) == (200, 20, 10)
    np.testing.assert_array_equal(cv2.imread(str(output)), image)
    classifier.assert_not_called()


def test_wrong_classifier_checkpoint_has_clear_diagnostic(example: ModuleType) -> None:
    image = np.zeros((12, 18, 3), dtype=np.uint8)
    result = detection_result(image, [[1, 2, 5, 6, 0.8, 1]])
    with pytest.raises(ValueError, match=r"no probabilities.*classification \(-cls\) checkpoint"):
        example.FilterClasses()(result)


@pytest.mark.parametrize("label", [
    "brown_bear", "American_black_bear", "ice_bear", "sloth_bear", "giant_panda", "lesser_panda",
])
def test_supported_classes_are_selected_even_outside_global_top1(example: ModuleType, label: str) -> None:
    image = np.zeros((12, 18, 3), dtype=np.uint8)
    result = Results(image, path="crop.jpg", names={0: "gibbon", 1: label, 2: "teddy_bear"}, probs=np.array([0.8, 0.15, 0.05]))

    filtered = example.FilterClasses((1,))(result)

    assert isinstance(filtered, Results)
    assert filtered.names == {0: label}
    assert filtered.probs.top1 == 0
    assert filtered.probs.top1conf == pytest.approx(0.15)
    assert result.names == {0: "gibbon", 1: label, 2: "teddy_bear"}
    np.testing.assert_array_equal(result.probs.data, [0.8, 0.15, 0.05])


def test_classifier_without_supported_classes_has_clear_diagnostic(example: ModuleType) -> None:
    image = np.zeros((12, 18, 3), dtype=np.uint8)
    result = Results(image, path="crop.jpg", names={0: "dog", 1: "teddy_bear"}, probs=np.array([0.8, 0.2]))

    with pytest.raises(ValueError, match="lacks the requested class IDs"):
        example.FilterClasses((151,))(result)


def test_empty_class_subset_is_rejected(example: ModuleType) -> None:
    with pytest.raises(ValueError, match="at least one classification class ID"):
        example.FilterClasses(())


@pytest.mark.parametrize("tensor", [False, True])
def test_class_filter_preserves_scores_and_source_without_renormalizing(example: ModuleType, tensor: bool) -> None:
    image = np.zeros((12, 18, 3), dtype=np.uint8)
    scores = np.array([0.8, 0.15, 0.05], dtype=np.float32)
    if tensor:
        import torch

        scores = torch.from_numpy(scores)
    result = Results(image, path="crop.jpg", names={0: "unrelated", 1: "first_breed", 2: "second_breed"}, probs=scores)

    filtered = example.FilterClasses((2, 1))(result)

    assert isinstance(filtered, Results)
    assert filtered.orig_img is image and filtered.path == result.path
    assert type(filtered.probs.data) is type(scores)
    assert filtered.names == {0: "second_breed", 1: "first_breed"}
    np.testing.assert_allclose(filtered.probs.cpu().numpy().data, [0.05, 0.15])
    assert filtered.probs.top1 == 1
    assert example.LabelDetections._classification_label(filtered) == "first breed / second breed"
    filtered.probs.data[0] = 0
    filtered.names[0] = "changed"
    np.testing.assert_allclose(result.probs.cpu().numpy().data, [0.8, 0.15, 0.05])
    assert result.names == {0: "unrelated", 1: "first_breed", 2: "second_breed"}


def test_class_filter_considers_candidates_outside_global_top_five(example: ModuleType) -> None:
    image = np.zeros((12, 18, 3), dtype=np.uint8)
    result = Results(
        image, path="crop.jpg", names={index: f"class_{index}" for index in range(8)},
        probs=np.array([0.4, 0.2, 0.15, 0.1, 0.07, 0.04, 0.03, 0.01]),
    )

    filtered = example.FilterClasses((6, 7))(result)

    assert not {6, 7} & set(result.probs.top5)
    assert filtered.names[filtered.probs.top1] == "class_6"
    assert filtered.probs.top1conf == pytest.approx(0.03)


def test_class_filter_retains_zero_score_candidates(example: ModuleType) -> None:
    image = np.zeros((12, 18, 3), dtype=np.uint8)
    result = Results(image, path="crop.jpg", names={0: "unrelated", 1: "first_breed"}, probs=np.array([1.0, 0.0]))

    filtered = example.FilterClasses((1,))(result)

    assert filtered.names == {0: "first_breed"}
    assert filtered.probs.top1conf == 0.0
    assert example.LabelDetections._classification_label(filtered) == "first breed"


def test_unrestricted_classes_preserve_native_ranking(example: ModuleType) -> None:
    image = np.zeros((12, 18, 3), dtype=np.uint8)
    result = Results(image, path="crop.jpg", names={0: "dog", 1: "teddy_bear"}, probs=np.array([0.2, 0.8]))

    filtered = example.FilterClasses()(result)

    assert filtered.names == result.names
    assert filtered.probs.top5 == result.probs.top5
    np.testing.assert_array_equal(filtered.probs.data, result.probs.data)


def test_pipeline_can_use_all_classifier_classes(example: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    detector = create_autospec(Model, instance=True, spec_set=True)
    detector.return_value = [detection_result(image, [[1, 2, 25, 28, 0.8, 1]])]
    classifier = classifier_model([[0.05, 0.7, 0.1, 0.03, 0.07, 0.03, 0.01, 0.01]])
    plotted = []

    def plot(result, **options):
        plotted.append(result)
        return image.copy()

    monkeypatch.setattr(Results, "plot", plot)
    example.build_pipeline(
        detector, classifier, output=tmp_path / "annotated.jpg", class_name="bear", classification_classes=None,
    )(image)

    assert plotted[0].names == {0: "teddy bear"}
    classifier.assert_called_once()


@pytest.mark.parametrize("has_bear", [False, True])
def test_cli_saves_only_the_annotated_image(
    example: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], has_bear: bool,
) -> None:
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    source = tmp_path / "source.png"
    assert cv2.imwrite(str(source), image)
    original_bytes = source.read_bytes()
    detector = create_autospec(Model, instance=True, spec_set=True)
    detector.return_value = [detection_result(image, [[1, 2, 25, 28, 0.8, 1]] if has_bear else [])]
    classifier = classifier_model([[0.05, 0.6, 0.1, 0.13, 0.07, 0.03, 0.01, 0.01]] if has_bear else [])
    factory = create_autospec(yolo.YOLO, spec_set=True, side_effect=[detector, classifier])
    monkeypatch.setattr(yolo, "YOLO", factory)
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [
        "run_detect_and_classify.py", "--source", str(source), "--output-dir", str(output),
        "--class-name", "bear", "--classification-classes", *map(str, BEAR_CLASS_IDS),
    ])

    assert example.main() == 0

    assert [call.kwargs["model"] for call in factory.call_args_list] == ["yolo26n.pt", "yolo26n-cls.pt"]
    assert {path.name for path in output.iterdir()} == {"annotated.jpg"}
    annotated = cv2.imread(str(output / "annotated.jpg"))
    assert annotated.shape == image.shape
    assert source.read_bytes() == original_bytes
    if has_bear:
        classifier.assert_called_once()
        assert classifier.call_args.kwargs["source"].shape == (26, 24, 3)
        assert annotated.any()
    else:
        classifier.assert_not_called()
        np.testing.assert_array_equal(cv2.imread(str(output / "annotated.jpg")), image)
    assert f"Saved {output / 'annotated.jpg'}" in capsys.readouterr().out


def test_cli_defaults_to_tutorial_image_and_dog_breeds(
    example: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert example.TUTORIAL_IMAGE.is_file()
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    source = tmp_path / "dogs.png"
    assert cv2.imwrite(str(source), image)
    monkeypatch.setattr(example, "TUTORIAL_IMAGE", source)
    detector = create_autospec(Model, instance=True, spec_set=True)
    detector.return_value = [detection_result(image, [[1, 2, 25, 28, 0.8, 0]])]
    classifier = create_autospec(Model, instance=True, spec_set=True)
    names = {index: f"breed-{index}" for index in range(151, 269)}
    scores = np.zeros(269, dtype=np.float32)
    scores[151], scores[207] = 0.6, 0.4
    classifier.return_value = [Results(image, path="crop.jpg", names=names, probs=scores)]
    factory = create_autospec(yolo.YOLO, spec_set=True, side_effect=[detector, classifier])
    monkeypatch.setattr(yolo, "YOLO", factory)
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["run_detect_and_classify.py", "--output-dir", str(output)])

    assert example.main() == 0

    assert {path.name for path in output.iterdir()} == {"annotated.jpg"}
    assert cv2.imread(str(output / "annotated.jpg")).shape == image.shape
    np.testing.assert_array_equal(classifier.return_value[0].probs.data, scores)
    classifier.assert_called_once()


@pytest.mark.parametrize("confidence,expected", [
    (0.49, "first breed / second breed"),
    (0.5, "first breed"),
    (0.8, "first breed"),
])
def test_classification_label_adds_runner_up_only_below_half(
    example: ModuleType, confidence: float, expected: str,
) -> None:
    classification = Results(
        np.zeros((12, 18, 3), dtype=np.uint8), path="crop.jpg",
        names={0: "first_breed", 1: "second_breed"}, probs=np.array([confidence, 0.2]),
    )

    assert example.LabelDetections._classification_label(classification) == expected


def test_annotation_uses_breeds_instead_of_numbers_and_applies_score_threshold(
    example: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((240, 360, 3), dtype=np.uint8)
    result = detection_result(image, [[200, 30, 300, 220, 0.8, 1], [1, 30, 100, 220, 0.7, 1]])
    detections, crops = example.CropResults()(example.FilterResults("bear")(result))
    classifications = [
        Results(crop, path="crop.jpg", names={0: "first_breed", 1: "second_breed", 2: "third_breed"}, probs=np.array(scores))
        for crop, scores in zip(crops, ([0.6, 0.3, 0.1], [0.4, 0.35, 0.25]))
    ]
    put_text = Mock(wraps=cv2.putText)
    monkeypatch.setattr(cv2, "putText", put_text)

    labeled = example.LabelDetections()(detections, classifications)
    annotated = results.Plot(conf=False, line_width=2)(labeled)

    labels = " ".join(call.args[1] for call in put_text.call_args_list)
    assert "#" not in labels
    assert "legend" not in labels
    assert "first breed" in labels
    assert "first breed / second breed" in labels
    assert labels.count("second breed") == 1
    assert labeled.names == {
        0: "first breed",
        1: "first breed / second breed",
    }
    assert labeled.boxes.cls.tolist() == [0, 1]
    assert detections.names == DETECTOR_NAMES
    assert detections.boxes.cls.tolist() == [1, 1]
    assert not np.shares_memory(labeled.boxes.data, detections.boxes.data)
    assert annotated.shape == image.shape
    assert all(call.args[4] == 2 / 3 for call in put_text.call_args_list)
    np.testing.assert_array_equal(annotated[230:], image[230:])
    assert not image.any()


def test_tutorial_before_and_after_images_have_identical_dimensions() -> None:
    assets = EXAMPLES.parent / "docs/assets/detect_and_classify"
    before = cv2.imread(str(assets / "dogs.png"))
    after = cv2.imread(str(assets / "annotated.jpg"))

    assert before is not None and after is not None
    assert before.shape == after.shape


@pytest.mark.parametrize("has_dog", [False, True])
def test_tutorial_snippets_run_in_order_without_example_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, has_dog: bool,
) -> None:
    tutorial = EXAMPLES.parent / "docs/tutorials/detect_and_classify.md"
    snippets = re.findall(r"```python[^\n]*\n(.*?)\n```", tutorial.read_text(), re.DOTALL)
    assert sum("pipeline = Pipeline(" in snippet for snippet in snippets) == 6
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "dogs.png"), image)
    monkeypatch.chdir(tmp_path)
    detector = create_autospec(Model, instance=True, spec_set=True)
    detector.return_value = [detection_result(image, [[1, 2, 25, 28, 0.8, 0]] if has_dog else [])]
    classifier = create_autospec(Model, instance=True, spec_set=True)
    names = {index: f"breed-{index}" for index in range(151, 269)}
    names[0] = "unrelated-class"
    scores = np.zeros(269, dtype=np.float32)
    scores[0], scores[151], scores[207] = 0.5, 0.3, 0.2
    classifier.return_value = [Results(image, path="crop.jpg", names=names, probs=scores)]
    factory = create_autospec(yolo.YOLO, spec_set=True, side_effect=[detector, detector, detector, detector, classifier, detector, classifier, detector, classifier])
    monkeypatch.setattr(yolo, "YOLO", factory)
    plotted = []
    native_plot = Results.plot

    def plot(result, **options):
        plotted.append(result)
        assert options["save"] is True
        assert options["filename"] == "annotated.jpg"
        return native_plot(result, **options)

    monkeypatch.setattr(Results, "plot", plot)
    module = ModuleType("detect_and_classify_tutorial")
    monkeypatch.setitem(sys.modules, module.__name__, module)

    exec(compile("\n\n".join(snippets), str(tutorial), "exec"), module.__dict__)

    assert len(plotted) == 1
    assert len(plotted[0].boxes) == int(has_dog)
    assert cv2.imread(str(tmp_path / "annotated.jpg")).shape == image.shape
    detector.assert_called_once()
    assert [call.kwargs["model"] for call in factory.call_args_list] == [
        "yolo26n.pt", "yolo26n.pt", "yolo26n.pt", "yolo26n.pt", "yolo26n-cls.pt",
        "yolo26n.pt", "yolo26n-cls.pt", "yolo26n.pt", "yolo26n-cls.pt",
    ]
    if has_dog:
        classifier.assert_called_once()
        filtered = module.FilterClasses(module.CLASSIFICATION_CLASSES)(classifier.return_value[0])
        assert filtered.names[filtered.probs.top1] == "breed-151"
        assert filtered.probs.top1conf == pytest.approx(0.3)
        assert plotted[0].names == {
            0: "breed-151 / breed-207",
        }
    else:
        classifier.assert_not_called()


def test_tutorial_pipeline_stages_preserve_operators_and_highlight_only_new_ones() -> None:
    tutorial = (EXAMPLES.parent / "docs/tutorials/detect_and_classify.md").read_text()
    snippets = re.findall(r"```python([^\n]*)\n(.*?)\n```", tutorial, re.DOTALL)
    stages = [(attributes.strip(), snippet) for attributes, snippet in snippets if "pipeline = Pipeline(" in snippet]
    assert len(stages) == 6
    previous: list[str] = []
    for attributes, snippet in stages:
        tree = ast.parse(snippet)
        assignment = next(
            statement for statement in tree.body
            if isinstance(statement, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "pipeline" for target in statement.targets)
        )
        nodes = assignment.value.args[0].elts
        operators = [ast.dump(value) for value in nodes]
        assert isinstance(nodes[0], ast.Call)
        cursor, added = 0, []
        for node, operator in zip(nodes, operators):
            if cursor < len(previous) and operator == previous[cursor]:
                cursor += 1
            else:
                added.append(node)
        assert cursor == len(previous)
        assert len(operators) > len(previous)
        if previous:
            highlights = re.fullmatch(r'hl_lines="([^"]+)"', attributes)
            assert highlights is not None
            highlighted = set()
            for span in highlights[1].split():
                bounds = [int(value) for value in span.split("-")]
                highlighted.update(range(bounds[0], bounds[-1] + 1))
            assert highlighted == {line for node in added for line in range(node.lineno, node.end_lineno + 1)}
        previous = operators
    assert len(previous) == 14
    before_run, run_section = tutorial.split("## Run it on an image", maxsplit=1)
    assert "pipeline(image)" not in before_run
    assert "cv2.imread" not in before_run
    assert "pipeline(image)" in run_section
    assert "cv2.imwrite" not in tutorial
    assert "to_bgr" not in tutorial
    assert 'Store("classified")' not in tutorial
    assert 'Recall("classified")' not in tutorial
    assert "@dataclass" not in tutorial
    classification_section, bonus_and_show = tutorial.split("## Classify objects", maxsplit=1)[1].split("## Bonus: Filter classification classes", maxsplit=1)
    bonus_section, show_section = bonus_and_show.split("## Show predictions", maxsplit=1)
    assert "https://ml-pipes.com/regions/#scatter-gather" in classification_section
    assert "Pick(1),  # Pass only the crops to the classifier." in classification_section
    assert "Recall" not in classification_section
    assert "FilterClasses" not in classification_section
    assert "class FilterClasses" in bonus_section
    assert "FilterClasses(CLASSIFICATION_CLASSES)" in bonus_section
    assert 'Recall("detections", prepend=True)' in show_section
    assert "LabelDetections()," in show_section
    assert show_section.split("## Run it on an image", maxsplit=1)[0].count("```python") == 1


def test_tutorial_opens_with_before_after_comparison_and_generic_stage_titles() -> None:
    tutorial = (EXAMPLES.parent / "docs/tutorials/detect_and_classify.md").read_text()
    intro, body = tutorial.split("## Detect objects", maxsplit=1)
    assert "| Before | After |" in intro
    assert "../assets/detect_and_classify/dogs.png" in intro
    assert "../assets/detect_and_classify/annotated.jpg" in intro
    assert "## Filter detections" in body
    assert "## Crop selected detections" in body
    assert body.index("## Filter detections") < body.index("## Crop selected detections")
    assert "## Classify objects" in body
    assert "detector = yolo.Predict" not in tutorial


def test_tutorial_operators_match_the_current_example() -> None:
    example_tree = ast.parse((EXAMPLES / "run_detect_and_classify.py").read_text())
    tutorial = (EXAMPLES.parent / "docs/tutorials/detect_and_classify.md").read_text()
    snippets = re.findall(r"```python[^\n]*\n(.*?)\n```", tutorial, re.DOTALL)
    tutorial_tree = ast.parse("\n\n".join(snippets))
    example_operators = {statement.name: statement for statement in example_tree.body if isinstance(statement, ast.ClassDef)}
    tutorial_operators = {statement.name: statement for statement in tutorial_tree.body if isinstance(statement, ast.ClassDef)}

    assert example_operators.keys() == tutorial_operators.keys()
    for name in example_operators:
        assert ast.dump(example_operators[name]) == ast.dump(tutorial_operators[name]), name


def test_unreadable_source_fails_before_model_construction(example: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    factory = Mock()
    monkeypatch.setattr(yolo, "YOLO", factory)
    monkeypatch.setattr(sys, "argv", ["run_detect_and_classify.py", "--source", str(tmp_path / "missing.jpg")])

    with pytest.raises(SystemExit) as raised:
        example.main()

    assert raised.value.code == 2
    factory.assert_not_called()


def test_cli_rejects_source_overwrite_before_native_plot_or_model_loading(
    example: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    source = tmp_path / "annotated.jpg"
    assert cv2.imwrite(str(source), image)
    original_bytes = source.read_bytes()
    factory = Mock()
    monkeypatch.setattr(yolo, "YOLO", factory)
    monkeypatch.setattr(sys, "argv", [
        "run_detect_and_classify.py", "--source", str(source), "--output-dir", str(tmp_path),
    ])

    with pytest.raises(SystemExit) as raised:
        example.main()

    assert raised.value.code == 2
    factory.assert_not_called()
    assert source.read_bytes() == original_bytes


def test_classification_helpers_live_inside_their_operator() -> None:
    for path in (EXAMPLES / "run_detect_and_classify.py", EXAMPLES.parent / "docs/tutorials/detect_and_classify.md"):
        source = path.read_text()
        if path.suffix == ".md":
            source = "\n\n".join(re.findall(r"```python[^\n]*\n(.*?)\n```", source, re.DOTALL))
        tree = ast.parse(source)
        module_functions = {statement.name for statement in tree.body if isinstance(statement, ast.FunctionDef)}
        operator = next(statement for statement in tree.body if isinstance(statement, ast.ClassDef) and statement.name == "LabelDetections")
        methods = {statement.name for statement in operator.body if isinstance(statement, ast.FunctionDef)}

        assert "_classification_label" in methods
        assert "_top_predictions" not in methods
        assert not {"top_predictions", "classification_label", "_top_predictions", "_classification_label"} & module_functions


def test_example_pipeline_ends_at_plot_without_classified_storage() -> None:
    source = (EXAMPLES / "run_detect_and_classify.py").read_text()
    tree = ast.parse(source)
    build_pipeline = next(statement for statement in tree.body if isinstance(statement, ast.FunctionDef) and statement.name == "build_pipeline")
    pipeline_return = next(statement for statement in build_pipeline.body if isinstance(statement, ast.Return))
    operators = pipeline_return.value.args[0].elts

    assert len(operators) == 14
    assert ast.unparse(operators[2].func) == "FilterResults"
    assert ast.unparse(operators[3].func) == "CropResults"
    assert ast.unparse(operators[9].func) == "FilterClasses"
    assert ast.unparse(operators[10].func) == "Gather"
    assert ast.unparse(operators[12]) == "LabelDetections()"
    assert isinstance(operators[-1], ast.Call)
    assert ast.unparse(operators[-1].func) == "results.Plot"
    assert 'Store("classified")' not in source
    assert 'Recall("classified")' not in source
    assert "crop_detections" not in source
    assert "Map(" not in source
