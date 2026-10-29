"""Exercise native segmentation geometry and PNG exports without model inference."""
from __future__ import annotations

import importlib
import re
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, create_autospec

import cv2
import numpy as np
import pytest
import torch
from ultralytics.engine.model import Model
from ultralytics.engine.results import Results

from ml_pipes.ultralytics import yolo

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
@pytest.mark.parametrize("segmentation", ["numpy", "tensor"], indirect=True)
def test_png_pixels_follow_original_coordinates(
    example: ModuleType, segmentation: Results, tmp_path: Path, crop: bool, background: str
) -> None:
    original = segmentation.orig_img.copy()
    original_masks = segmentation.masks.cpu().numpy().data.copy()
    paths = example.IsolateObjects(
        tmp_path, source_name="scene", crop=crop, background=background
    )(segmentation)

    assert len(paths) == 2
    for index, (path, region) in enumerate(zip(paths, [(1, 5, 2, 6), (1, 4, 8, 11)])):
        y0, y1, x0, x1 = region
        mask = np.zeros((6, 12), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 255
        expected = np.where(mask[:, :, None] != 0, original, 0)
        if background == "transparent":
            expected = np.dstack((expected, mask))
        if crop:
            expected = expected[y0:y1, x0:x1]
        actual = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(actual, expected)
        assert re.fullmatch(rf"scene_[0-9a-f]{{32}}_object-{index:03d}_class-0-person.png", path.name)
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
    paths = example.IsolateObjects(tmp_path)(segmentation)
    assert len(paths) == 2
    expected_masks = np.zeros((2, 6, 12), dtype=np.uint8)
    expected_masks[0, 1:5, 2:6] = 255
    expected_masks[1, 1:4, 8:11] = 255
    for path, mask in zip(paths, expected_masks):
        output = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(output[:, :, 3], mask)


def test_crop_rounds_outward_and_clips_all_image_edges(
    example: ModuleType, segmentation: Results, tmp_path: Path
) -> None:
    segmentation = segmentation.numpy()
    segmentation.boxes.data[:, :4] = [[-3.2, 0.2, 6.3, 9.8], [7.8, -2.3, 20.2, 3.2]]
    paths = example.IsolateObjects(tmp_path, crop=True)(segmentation)
    first, second = [cv2.imread(str(path), cv2.IMREAD_UNCHANGED) for path in paths]
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
    assert example.IsolateObjects(tmp_path, crop=True)(segmentation) == []
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
    assert example.IsolateObjects(output)(segmentation) == []
    assert output.is_dir()
    assert list(output.iterdir()) == []


def test_repeated_calls_and_same_class_objects_keep_unique_safe_paths(
    example: ModuleType, segmentation: Results, tmp_path: Path
) -> None:
    segmentation.names[0] = "../person/with spaces"
    isolate = example.IsolateObjects(tmp_path, source_name="../same scene")
    first = isolate(segmentation)
    saved = [path.read_bytes() for path in first]
    second = isolate(segmentation)
    third = example.IsolateObjects(tmp_path, source_name="../same scene")(segmentation)
    assert len(set(first + second + third)) == 6
    assert [path.read_bytes() for path in first] == saved
    assert all(path.parent == tmp_path for path in first + second + third)
    assert all("class-0-person_with_spaces" in path.name for path in first)


def test_write_failure_is_reported(
    example: ModuleType, segmentation: Results, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cv2, "imwrite", Mock(return_value=False))
    with pytest.raises(OSError, match="Could not save"):
        example.IsolateObjects(tmp_path)(segmentation)


def test_invalid_background_is_rejected(example: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="background"):
        example.IsolateObjects(tmp_path, background="white")


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
    paths = pipeline(segmentation.orig_img)
    assert len(paths) == 2
    assert cv2.imread(str(paths[0]), cv2.IMREAD_UNCHANGED).shape == (4, 4, 3)
    factory.assert_called_once_with(model="custom-seg.pt", task=None, verbose=False)
    model.assert_called_once()
    assert model.call_args.kwargs["source"] is segmentation.orig_img
    assert {key: value for key, value in model.call_args.kwargs.items() if key != "source"} == {
        "classes": [0], "conf": 0.4, "iou": 0.6, "imgsz": 320, "device": "cpu", "max_det": 12,
    }
    model.track.assert_not_called()
    model.embed.assert_not_called()
