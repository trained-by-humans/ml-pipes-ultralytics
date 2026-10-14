"""Return the cosine similarity of two whole-image YOLO embeddings.

Run from the repository root:
    python examples/run_image_similarity.py
    python examples/run_image_similarity.py --image-b docs/assets/image_similarity/bus-1-flipped.png
    python examples/run_image_similarity.py --image-a first.jpg --image-b second.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from ml_pipes.core import Pipeline
from ml_pipes.operator import Operator
from ml_pipes.standard import Map
from ml_pipes.ultralytics import yolo

SAMPLE_IMAGES = Path(__file__).resolve().parents[1] / "docs" / "assets" / "image_similarity"


def require_two_images(sources: list[str]) -> list[str]:
    """Check that two local image files are available before inference."""
    if len(sources) != 2:
        raise ValueError("Expected exactly two image paths.")
    for source in sources:
        if not Path(source).is_file():
            raise ValueError(f"Image file does not exist: {source}")
    return sources


@Operator
class CosineSimilarity:
    """Reduce two native feature vectors to a cosine-similarity score."""

    def __call__(self, embeddings: list[Tensor]) -> float:
        if len(embeddings) != 2:
            raise ValueError("Expected exactly two embedding vectors.")
        if any(vector.ndim != 1 or vector.numel() == 0 for vector in embeddings):
            raise ValueError("Embeddings must be non-empty, one-dimensional vectors.")
        if embeddings[0].shape != embeddings[1].shape:
            raise ValueError("Embedding vectors must have the same shape.")

        features = torch.stack(embeddings).float()
        if not torch.isfinite(features).all().item():
            raise ValueError("Embedding vectors must contain only finite values.")
        norms = torch.linalg.vector_norm(features, dim=1)
        if not torch.isfinite(norms).all().item() or (norms == 0).any().item():
            raise ValueError("Embedding vectors must have finite, non-zero norms.")

        score = F.cosine_similarity(features[0], features[1], dim=0, eps=0.0)
        return score.clamp(-1.0, 1.0).item()


def build_pipeline(
    model: str = "yolo26n.pt",
    *,
    imgsz: int = 640,
    device: str | None = None,
) -> Pipeline[list[str], float]:
    """Build one reusable model followed by a cosine-similarity step."""
    return Pipeline(
        [
            Map(require_two_images),
            yolo.Embed(model=model, imgsz=imgsz, device=device, rect=False),
            CosineSimilarity(),
        ],
        auto_validate=True,
    )


def positive_int(value: str) -> int:
    """Require a positive image size at the CLI boundary."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image-a", type=Path, default=SAMPLE_IMAGES / "bus-1.png",
        help="First local image. Defaults to the tutorial's bus-1.png.",
    )
    parser.add_argument(
        "--image-b", type=Path, default=SAMPLE_IMAGES / "bus-2.png",
        help="Second local image. Defaults to the tutorial's bus-2.png.",
    )
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--imgsz", type=positive_int, default=640)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    sources = [str(args.image_a), str(args.image_b)]
    try:
        require_two_images(sources)
    except ValueError as error:
        parser.error(str(error))

    pipeline = build_pipeline(
        args.model,
        imgsz=args.imgsz,
        device=args.device,
    )
    pipeline.validate()
    print(f"Image A: {args.image_a.name}")
    print(f"Image B: {args.image_b.name}")
    similarity = pipeline(sources)
    print(f"Cosine similarity: {similarity:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
