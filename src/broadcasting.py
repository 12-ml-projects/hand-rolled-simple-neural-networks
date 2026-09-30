from src.custom_types import Array


def unbroadcast(adjoint: Array, shape: tuple[int, ...]) -> Array:
    """Reduce `adjoint` back to `shape`, undoing a broadcast made by numpy.

    Broadcasting copies a stretched operand, and the adjoint of a copy is a sum,
    so every entry an operand was expanded into contributes back to it.
    """
    if adjoint.shape == shape:
        return adjoint

    _require_broadcastable(adjoint.shape, shape)

    prepended = adjoint.ndim - len(shape)
    if prepended:
        adjoint = adjoint.sum(axis=tuple(range(prepended)))

    stretched = tuple(
        axis for axis, size in enumerate(shape) if size == 1 != adjoint.shape[axis]
    )
    if stretched:
        adjoint = adjoint.sum(axis=stretched, keepdims=True)

    return adjoint


def _require_broadcastable(
    adjoint_shape: tuple[int, ...], shape: tuple[int, ...]
) -> None:
    """A mismatch that is not a broadcast means the operator's backward is wrong."""
    prepended = len(adjoint_shape) - len(shape)
    aligned = adjoint_shape[prepended:] if prepended >= 0 else ()

    if prepended < 0 or any(
        size not in (1, actual) for size, actual in zip(shape, aligned)
    ):
        raise ValueError(
            f"An adjoint of shape {adjoint_shape} cannot belong to a value of "
            f"shape {shape}, which does not broadcast to it."
        )
