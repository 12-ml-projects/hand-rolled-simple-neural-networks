import gzip
import struct
import urllib.request
from pathlib import Path
from typing import TypeAlias

import numpy as np

from src.custom_types import Array

Split: TypeAlias = tuple[Array, Array]
Batch: TypeAlias = tuple[Array, Array]

CLASSES = 10
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
ARCHIVES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def load(directory: Path = Path("data")) -> tuple[Split, Split]:
    """MNIST as (train, test), each a pair of flat features and integer labels.

    Downloads on first use and caches the archives. Uses the dataset's own
    train/test split.
    """
    raw = {name: _fetch(directory, archive) for name, archive in ARCHIVES.items()}

    return (
        (_features(raw["train_images"]), _labels(raw["train_labels"])),
        (_features(raw["test_images"]), _labels(raw["test_labels"])),
    )


def batches(
    features: Array,
    labels: Array,
    batch_size: int,
    rng: np.random.Generator | None = None,
) -> list[Batch]:
    """Cut a split into batches, shuffled if given a generator.

    The last batch is kept even when short: dropping it would throw away data,
    and the loss is a mean so a smaller batch is not mis-weighted.
    """
    order = np.arange(len(labels)) if rng is None else rng.permutation(len(labels))

    return [
        (features[chunk], labels[chunk])
        for chunk in (
            order[start : start + batch_size]
            for start in range(0, len(order), batch_size)
        )
    ]


def _fetch(directory: Path, archive: str) -> bytes:
    path = directory / archive

    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MIRROR + archive, path)

    return gzip.decompress(path.read_bytes())


def _parse_idx(raw: bytes) -> Array:
    """IDX: a magic number whose low byte is the rank, then the dimensions."""
    (magic,) = struct.unpack(">I", raw[:4])
    rank = magic & 0xFF
    header = 4 + 4 * rank
    shape = struct.unpack(f">{rank}I", raw[4:header])

    return np.frombuffer(raw, dtype=np.uint8, offset=header).reshape(shape)


def _features(raw: bytes) -> Array:
    images = _parse_idx(raw)

    return images.reshape(len(images), -1).astype(np.float64) / 255.0


def _labels(raw: bytes) -> Array:
    return _parse_idx(raw).astype(np.int64)
