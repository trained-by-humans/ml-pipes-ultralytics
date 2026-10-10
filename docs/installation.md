---
title: Install ml-pipes-ultralytics
description: >-
  Set up a Python environment and install the Ultralytics YOLO operators and the core and vision dependencies used by the tutorials.
---

# Installation

Use Python 3.10 or newer. A virtual environment keeps the package and its
dependencies separate from other projects:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ml-pipes-ultralytics
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.
The package installs `ml-pipes-core`, `ml-pipes-vision`, and Ultralytics, along
with their dependencies, including NumPy, OpenCV, and PyTorch. The tutorials
do not require a separate vision-operator installation.

If working from a clone of this repository, install the checkout from its
root directory instead of the published package:

```bash
python -m pip install -e .
```

Model checkpoints such as `yolo26n.pt` may be downloaded when a model operator
is constructed. Sample-image downloads and environment/model setup should
be completed before measuring inference performance.

For GPU acceleration, use a PyTorch build appropriate for your hardware;
see the [PyTorch installation instructions](https://pytorch.org/get-started/locally/).
Configure the model operator's `device` option when selecting a particular
device. The guides also work on CPU; performance depends on the machine.

Ultralytics software and model weights have separate licensing terms. Read
the [licensing note](index.md#licensing) before using or redistributing them.
