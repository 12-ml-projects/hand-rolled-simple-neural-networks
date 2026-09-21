from collections.abc import Sequence
from typing import TypeAlias

Slope: TypeAlias = float
Intercept: TypeAlias = float


def least_squares_fit(
    xs: Sequence[float], ys: Sequence[float]
) -> tuple[Slope, Intercept]:
    if len(xs) != len(ys):
        raise ValueError("Need one y for every x.")

    count = len(xs)
    if count < 2:
        raise ValueError("Need at least two points to fit a line.")

    mean_x = sum(xs) / count
    mean_y = sum(ys) / count

    variance = sum((x - mean_x) ** 2 for x in xs)
    if variance == 0.0:
        raise ValueError("Cannot fit a line through points sharing a single x.")

    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = covariance / variance

    return slope, mean_y - slope * mean_x
