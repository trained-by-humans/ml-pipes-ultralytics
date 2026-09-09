# Contributing

Thanks for contributing to `ml-pipes-ultralytics`.
Honestly, just the fact that you opened this page makes me happy! ^^

## Install the repository

Clone the repository, create and activate a Python 3.10+ virtual environment,
then install the package with its development dependencies in editable mode:

```bash
git clone https://github.com/requiem4machines/ml-pipes-ultralytics.git
cd ml-pipes-ultralytics
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Tests

Run the test suite from the package root before submitting a change:

```bash
python -m pytest -q
```

Add or update tests for every behavior change. Tests must not require a model
download, network access, or a display server.

## Documentation

The public package catalog is in `docs/INDEX.md` and the upstream API coverage
comparison is in `../docs/COVERAGE.md`. Keep both current when adding, removing,
or changing a public operator. Examples in `examples/` should be runnable from
the repository root and demonstrate pipeline composition.
