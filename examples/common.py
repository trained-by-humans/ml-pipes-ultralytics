"""Local, ignored runtime assets shared by the runnable examples."""
from __future__ import annotations

from pathlib import Path
from shutil import copy2

from ultralytics.utils import ASSETS

EXAMPLE_ASSETS = Path(__file__).parent / ".example_assets"


def default_image() -> Path:
    """Return a local copy of Ultralytics' bundled ``bus.jpg`` example input."""
    image = EXAMPLE_ASSETS / "bus.jpg"
    if not image.exists():
        EXAMPLE_ASSETS.mkdir(parents=True, exist_ok=True)
        copy2(ASSETS / "bus.jpg", image)
    return image
