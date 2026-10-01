from collections.abc import Callable, Iterator, Sequence
from typing import TypeAlias

from src.custom_types import Array
from src.ffnn.optimiser import Optimiser
from src.tensor import Tensor

Batch: TypeAlias = tuple[Array, Array]
Model: TypeAlias = Callable[[Tensor], Tensor]
Criterion: TypeAlias = Callable[[Tensor, Array], Tensor]


def train(
    model: Model,
    batches: Sequence[Batch],
    criterion: Criterion,
    optimiser: Optimiser,
    epochs: int,
) -> Iterator[float]:
    for _ in range(epochs):
        total = 0.0

        for features, labels in batches:
            optimiser.zero_grad()

            loss = criterion(model(Tensor(features, requires_grad=False)), labels)
            loss.backward()

            optimiser.step()
            total += float(loss.value)

        yield total / len(batches)


def evaluate(model: Model, batches: Sequence[Batch], criterion: Criterion) -> float:
    total = 0.0

    for features, labels in batches:
        loss = criterion(model(Tensor(features, requires_grad=False)), labels)
        total += float(loss.value)

    return total / len(batches)
