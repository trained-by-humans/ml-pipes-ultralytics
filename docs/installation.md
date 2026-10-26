---
title: Install ml-pipes-ultralytics
description: >-
  Set up a Python environment and install the Ultralytics YOLO operators and the core and vision dependencies used by the tutorials.
---

# Installation

Use Python 3.10 or newer. If you need help setting up Python or a virtual
environment, see [Python setup](#python-setup) below.

## Install from PyPI

Install the latest published release:

```bash
python -m pip install ml-pipes-ultralytics
```

The package installs `ml-pipes-core`, `ml-pipes-vision`, and Ultralytics, along
with their dependencies, including NumPy, OpenCV, and PyTorch. The tutorials
do not require a separate vision-operator installation.

## Install the latest changes from GitHub

To use changes that have not yet been published to PyPI, install directly
from the repository's `main` branch. This requires [Git](https://git-scm.com/downloads):

```bash
python -m pip install --upgrade "git+https://github.com/trained-by-humans/ml-pipes-ultralytics.git@main"
```

## Get the tutorial files

You can follow the tutorial snippets in your own script or notebook. To run
the complete example scripts instead, you need a local repository checkout;
installing the package does not place these scripts in your working directory.
Clone the repository and change into its root directory:

```bash
git clone https://github.com/trained-by-humans/ml-pipes-ultralytics.git
cd ml-pipes-ultralytics
```

## Models and acceleration

Model checkpoints such as `yolo26n.pt` may be downloaded when a model operator
is constructed. Sample-image downloads and environment/model setup should
be completed before measuring inference performance.

For GPU acceleration, use a PyTorch build appropriate for your hardware;
see the [PyTorch installation instructions](https://pytorch.org/get-started/locally/).
Configure the model operator's `device` option when selecting a particular
device. The guides also work on CPU; performance depends on the machine.

## Python setup

If you are new to Python, install Python 3.10 or newer from
[python.org](https://www.python.org/downloads/).

A virtual environment keeps this package and its dependencies separate from
other projects. Open a terminal in your project directory (the repository root
if you have cloned it), check your Python version, then create and activate
an environment:

=== "macOS"

    ```bash hl_lines="3"
    python3 --version
    python3 -m venv .venv
    source .venv/bin/activate
    ```

=== "Linux"

    ```bash hl_lines="3"
    python3 --version
    python3 -m venv .venv
    source .venv/bin/activate
    ```

=== "Windows"

    ```bat hl_lines="3"
    python --version
    python -m venv .venv
    .venv\Scripts\activate
    ```

Then update pip, Python's package installer:

```bash
python -m pip install --upgrade pip
```

With the environment active, run one of the installation commands above.
Activate the same environment again whenever you open a new terminal to
work through a tutorial.
