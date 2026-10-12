# Contributing

Thanks for contributing to `ml-pipes-ultralytics`.
Honestly, just the fact that you opened this page makes me happy! ^^

## Install the repository

Clone the repository, create and activate a Python 3.10+ virtual environment,
then install the package in editable mode:

```bash
git clone https://github.com/trained-by-humans/ml-pipes-ultralytics.git
cd ml-pipes-ultralytics
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Tests

Install the test dependencies, then run the full suite from the repository
root before submitting a change:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

Add or update tests in `tests/` for behavior changes. Tests must not require a
model download, network access, or a display server. While developing, run a
single test module with:

```bash
python -m pytest tests/test_core.py -q
```

The suite may emit deprecation warnings from third-party dependencies; these
warnings do not currently fail the test run.

## Pull requests

Keep changes focused and include tests for behavior changes. Update
`docs/reference.md` and `docs/coverage.md` when a change adds or changes a
public Ultralytics operator.

## Documentation

Use lowercase filenames and directory names for documentation guides in
`docs/` and package-local documentation directories. Keep existing underscores
and conventional filenames such as `README.md`, `AGENTS.md`, and `SKILL.md`.
Visible page titles and navigation labels retain their normal capitalization.
Update Markdown links, package documentation URLs, and cross-site links when
renaming a guide; URLs follow source filename casing.

The `Pages` workflow tests the naming checks and builds the site on pull
requests and `main`, but deploys only from `main`. It checks filenames before
building and generated URLs afterward. To run these checks locally:

```bash
python .github/scripts/check_docs_paths.py
python -m mkdocs build --strict --site-dir _site
python .github/scripts/check_docs_paths.py --site-dir _site
```

Preview the documentation site locally from the package root:

```bash
python -m pip install -e '.[docs]'
python -m mkdocs serve
```

Open `http://127.0.0.1:8000` in a browser. Use the following command to
validate the static site before submitting documentation changes:

```bash
python -m mkdocs build --strict
```

GitHub Actions deploys documentation to GitHub Pages when changes are pushed
to `main`; contributors do not need to deploy it manually.
