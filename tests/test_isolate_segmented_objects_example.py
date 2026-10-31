"""Exercise native segmentation geometry and PNG exports without model inference."""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, create_autospec

import cv2
import numpy as np
import pytest
import torch
from ultralytics.engine.model import Model
from ultralytics.engine.results import Results

from ml_pipes.core import Pipeline
from ml_pipes.standard import Pick
from ml_pipes.ultralytics import results, yolo

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def example(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    return importlib.import_module("run_isolate_segmented_objects")


@pytest.fixture
def segmentation(request: pytest.FixtureRequest) -> Results:
    # A 6x12 original is letterboxed to 12x12: three padding rows per side.
    image = np.arange(6 * 12 * 3, dtype=np.uint8).reshape(6, 12, 3) + 1
    masks = np.zeros((2, 12, 12), dtype=np.uint8)
    masks[0, 4:8, 2:6] = 1
    masks[1, 4:7, 8:11] = 1
    boxes = np.array([[2, 1, 6, 5, 0.9, 0], [8, 1, 11, 4, 0.8, 0]], dtype=np.float32)
    tensor = getattr(request, "param", "numpy") == "tensor"
    return Results(
        orig_img=image,
        path="scene.jpg",
        names={0: "person"},
        boxes=torch.from_numpy(boxes) if tensor else boxes,
        masks=torch.from_numpy(masks) if tensor else masks,
    )


@pytest.fixture
def native_model(monkeypatch: pytest.MonkeyPatch, segmentation: Results) -> tuple[Mock, Mock]:
    model = create_autospec(Model, instance=True, spec_set=True)
    model.return_value = [segmentation]
    factory = create_autospec(yolo.YOLO, spec_set=True, return_value=model)
    monkeypatch.setattr(yolo, "YOLO", factory)
    return factory, model


@pytest.mark.parametrize("crop", [False, True])
@pytest.mark.parametrize("background", ["transparent", "black"])
@pytest.mark.parametrize("save", [False, True])
@pytest.mark.parametrize("segmentation", ["numpy", "tensor"], indirect=True)
def test_isolated_pixels_and_optional_pngs_follow_original_coordinates(
    example: ModuleType, segmentation: Results, tmp_path: Path,
    crop: bool, background: str, save: bool,
) -> None:
    original = segmentation.orig_img.copy()
    original_masks = segmentation.masks.cpu().numpy().data.copy()
    output = tmp_path / "objects"
    detections, images = example.IsolateObjects(
        output_dir=output, save=save, source_name="scene", crop=crop, background=background
    )(segmentation)

    assert len(detections.boxes) == len(images) == 2
    np.testing.assert_array_equal(detections.boxes.cpu().numpy().data, segmentation.boxes.cpu().numpy().data)
    paths = sorted(output.glob("*.png"))
    assert len(paths) == (2 if save else 0)
    assert output.exists() is save
    for index, (image, region) in enumerate(zip(images, [(1, 5, 2, 6), (1, 4, 8, 11)])):
        y0, y1, x0, x1 = region
        mask = np.zeros((6, 12), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 255
        expected = np.where(mask[:, :, None] != 0, original, 0)
        if background == "transparent":
            expected = np.dstack((expected, mask))
        if crop:
            expected = expected[y0:y1, x0:x1]
        np.testing.assert_array_equal(image, expected)
        assert not np.shares_memory(image, segmentation.orig_img)
        if save:
            np.testing.assert_array_equal(cv2.imread(str(paths[index]), cv2.IMREAD_UNCHANGED), image)
            assert re.fullmatch(rf"scene_[0-9a-f]{{32}}_object-{index:03d}_class-0-person.png", paths[index].name)
    np.testing.assert_array_equal(segmentation.orig_img, original)
    np.testing.assert_array_equal(segmentation.masks.data, original_masks)


@pytest.mark.parametrize("mask_size", ["original", "resized-and-padded"])
def test_mask_geometry_supports_original_and_resized_masks(
    example: ModuleType, segmentation: Results, tmp_path: Path, mask_size: str
) -> None:
    if mask_size == "original":
        masks = np.zeros((2, 6, 12), dtype=np.uint8)
        masks[0, 1:5, 2:6] = 1
        masks[1, 1:4, 8:11] = 1
    else:
        # Double the input width and height, then add six padding rows per side.
        masks = np.zeros((2, 24, 24), dtype=np.uint8)
        masks[0, 8:15, 4:11] = 1
        masks[1, 8:13, 16:21] = 1
    segmentation.update(masks=masks)
    detections, images = example.IsolateObjects(output_dir=tmp_path / "unused")(segmentation)
    assert len(detections.boxes) == len(images) == 2
    expected_masks = np.zeros((2, 6, 12), dtype=np.uint8)
    expected_masks[0, 1:5, 2:6] = 255
    expected_masks[1, 1:4, 8:11] = 255
    for image, mask in zip(images, expected_masks):
        np.testing.assert_array_equal(image[:, :, 3], mask)


def test_crop_rounds_outward_and_clips_all_image_edges(
    example: ModuleType, segmentation: Results, tmp_path: Path
) -> None:
    segmentation = segmentation.numpy()
    segmentation.boxes.data[:, :4] = [[-3.2, 0.2, 6.3, 9.8], [7.8, -2.3, 20.2, 3.2]]
    detections, images = example.IsolateObjects(output_dir=tmp_path / "unused", crop=True)(segmentation)
    assert len(detections.boxes) == len(images) == 2
    first, second = images
    assert first.shape == (6, 7, 4)
    assert second.shape == (4, 5, 4)
    np.testing.assert_array_equal(first[1:5, 2:6, :3], segmentation.orig_img[1:5, 2:6])
    np.testing.assert_array_equal(second[1:4, 1:4, :3], segmentation.orig_img[1:4, 8:11])


@pytest.mark.parametrize("box", [[20, 1, 25, 4], [-8, 1, -2, 4], [5, 3, 5, 3], [8, 4, 2, 1]])
def test_invalid_crop_regions_are_skipped(
    example: ModuleType, segmentation: Results, tmp_path: Path, box: list[int]
) -> None:
    segmentation = segmentation.numpy()
    segmentation.boxes.data[:, :4] = box
    detections, images = example.IsolateObjects(output_dir=tmp_path, save=True, crop=True)(segmentation)
    assert len(detections.boxes) == len(detections.masks) == 0
    assert images == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("case", ["no-masks", "no-boxes", "empty-detections", "empty-polygons"])
def test_missing_or_empty_results_produce_no_files(
    example: ModuleType, segmentation: Results, tmp_path: Path, case: str
) -> None:
    if case == "no-masks":
        segmentation.masks = None
    elif case == "no-boxes":
        segmentation.boxes = None
    elif case == "empty-detections":
        segmentation = segmentation[:0]
    else:
        segmentation.masks.data[:] = 0
    output = tmp_path / "objects"
    detections, images = example.IsolateObjects(output_dir=output, save=True)(segmentation)
    assert detections.boxes is None or len(detections.boxes) == 0
    assert detections.masks is None or len(detections.masks) == 0
    assert images == []
    assert output.is_dir()
    assert list(output.iterdir()) == []


def test_repeated_calls_and_same_class_objects_keep_unique_safe_paths(
    example: ModuleType, segmentation: Results, tmp_path: Path
) -> None:
    segmentation.names[0] = "../person/with spaces"
    isolate = example.IsolateObjects(output_dir=tmp_path, save=True, source_name="../same scene")
    isolate(segmentation)
    first = set(tmp_path.glob("*.png"))
    saved = {path: path.read_bytes() for path in first}
    isolate(segmentation)
    second = set(tmp_path.glob("*.png")) - first
    example.IsolateObjects(output_dir=tmp_path, save=True, source_name="../same scene")(segmentation)
    third = set(tmp_path.glob("*.png")) - first - second
    assert len(first) == len(second) == len(third) == 2
    assert {path: path.read_bytes() for path in first} == saved
    assert all(path.parent == tmp_path for path in first | second | third)
    assert all("class-0-person_with_spaces" in path.name for path in first)


def test_write_failure_is_reported(
    example: ModuleType, segmentation: Results, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cv2, "imwrite", Mock(return_value=False))
    with pytest.raises(OSError, match="Could not save"):
        example.IsolateObjects(output_dir=tmp_path, save=True)(segmentation)


def test_invalid_background_is_rejected(example: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="background"):
        example.IsolateObjects(output_dir=tmp_path, background="white")


@pytest.mark.parametrize("skip", ["empty-mask", "invalid-box"])
def test_skipped_objects_keep_detections_and_images_aligned(
    example: ModuleType, segmentation: Results, tmp_path: Path, skip: str
) -> None:
    if skip == "empty-mask":
        segmentation.masks.data[0] = 0
    else:
        segmentation.boxes.data[0, :4] = [5, 3, 5, 3]
    original_boxes = segmentation.boxes.data.copy()
    detections, images = example.IsolateObjects(output_dir=tmp_path / "unused", crop=True)(segmentation)

    assert len(detections.boxes) == len(detections.masks) == len(images) == 1
    np.testing.assert_array_equal(detections.boxes.data, original_boxes[1:])
    np.testing.assert_array_equal(detections.masks.data, segmentation.masks.data[1:])
    np.testing.assert_array_equal(images[0][:, :, :3], segmentation.orig_img[1:4, 8:11])
    np.testing.assert_array_equal(segmentation.boxes.data, original_boxes)
    assert len(segmentation.boxes) == 2


@pytest.mark.parametrize("save", [False, True])
def test_cli_save_flag_controls_png_writing(
    example: ModuleType, native_model: tuple[Mock, Mock], segmentation: Results,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, save: bool
) -> None:
    source = tmp_path / "scene.png"
    assert cv2.imwrite(str(source), segmentation.orig_img)
    output = tmp_path / "objects"
    monkeypatch.setattr(example, "default_image", lambda: source)
    argv = ["run_isolate_segmented_objects.py", "--output-dir", str(output), "--crop"]
    if save:
        argv.append("--save")
    monkeypatch.setattr(sys, "argv", argv)

    assert example.main() == 0

    assert output.exists() is save
    assert len(list(output.glob("*.png"))) == (2 if save else 0)


@pytest.mark.parametrize("segmentation", ["tensor"], indirect=True)
def test_pipeline_passes_native_results_from_predict(
    example: ModuleType, native_model: tuple[Mock, Mock], segmentation: Results, tmp_path: Path
) -> None:
    factory, model = native_model
    pipeline = example.build_pipeline(
        "custom-seg.pt", tmp_path, source_name="scene", classes=[0], conf=0.4,
        iou=0.6, imgsz=320, device="cpu", max_det=12, crop=True, background="black",
    )
    pipeline.validate()
    detections, images = pipeline(segmentation.orig_img)
    assert len(detections.boxes) == len(images) == 2
    assert images[0].shape == (4, 4, 3)
    np.testing.assert_array_equal(detections.boxes.cpu().numpy().data, segmentation.boxes.cpu().numpy().data)
    assert list(tmp_path.iterdir()) == []
    factory.assert_called_once_with(model="custom-seg.pt", task=None, verbose=False)
    model.assert_called_once()
    assert model.call_args.kwargs["source"] is segmentation.orig_img
    assert {key: value for key, value in model.call_args.kwargs.items() if key != "source"} == {
        "classes": [0], "conf": 0.4, "iou": 0.6, "imgsz": 320, "device": "cpu", "max_det": 12,
    }
    model.track.assert_not_called()
    model.embed.assert_not_called()


def test_isolated_detections_can_be_used_by_native_result_operators(
    example: ModuleType, segmentation: Results, tmp_path: Path
) -> None:
    pipeline = Pipeline(
        [example.IsolateObjects(output_dir=tmp_path / "unused"), Pick(0), results.Summary()],
        auto_validate=True,
    )
    pipeline.validate(Results)

    summary = pipeline(segmentation)

    assert len(summary) == 2
    assert all(item["name"] == "person" for item in summary)
    assert list(tmp_path.iterdir()) == []
