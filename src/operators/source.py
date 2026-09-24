from src.custom_types import Value

from .operator import Operator


class Source(Operator):
    def forward(self, x: Value) -> Value:  # type: ignore[override]
        return x

    def backward(  # type: ignore[override]
        self, adjoint: Value, x: Value
    ) -> tuple[Value, ...]:
        return ()
