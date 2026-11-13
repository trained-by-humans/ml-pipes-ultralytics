# Changelog

## 0.1.5

### Changed

- Raise the Ultralytics baseline from `8.4.174` to `8.4.212` and add an
  upper bound of `<8.5` so the next minor series is reviewed before use.

[Full comparison with 0.1.4](https://github.com/trained-by-humans/ml-pipes-ultralytics/compare/v0.1.4...v0.1.5)

## 0.1.4

### Fixed

- Importing `ml_pipes.ultralytics` now registers BGR image previews for bare
  NumPy arrays, matching Ultralytics and OpenCV. Inspection converts previews
  to RGB without changing source arrays. This changes the global default for
  NumPy image previews; pipelines using RGB arrays can register an RGB
  formatter on their inspector as described in the documentation reference.
  Non-image arrays retain their shape and dtype summaries.

### Added

- A detect-and-classify tutorial and runnable example that detect objects,
  crop them, classify each crop, and annotate the original detections.
- A segmented-object isolation tutorial and runnable example with in-memory
  images, transparent or black backgrounds, optional cropping, and PNG saving.
- An image-similarity tutorial and runnable example using whole-image YOLO
  embeddings and cosine similarity.
- Python 3.13 in the package classifiers, matching the existing CI coverage.

### Changed

- Require stable `ml-pipes-core` and `ml-pipes-vision` versions
  `>=0.1.3,<0.2.0` instead of exact `0.1.2rc1` pins. Version `0.1.3` is the
  tested framework baseline; the next minor release requires compatibility
  testing before extending the bounds.
- Clarify installation and tutorial setup, upgrade the documentation theme
  to Pixel Lab 0.2.0, and add page source links.

The prediction, embedding, tracking, and result-processing operator APIs are
unchanged. The new example operators remain in the repository's `examples/`
directory and are not installed as package APIs.

[Full comparison with 0.1.3](https://github.com/trained-by-humans/ml-pipes-ultralytics/compare/v0.1.3...v0.1.4)
