from src.custom_types import Value

from .operator import Operator


class Add(Operator):
    def forward(self, x: Value, y: Value) -> Value:  # type: ignore[override]
        return x + y

    def backward(  # type: ignore[override]
        self, adjoint: Value, x: Value, y: Value
    ) -> tuple[Value, Value]:
        return (adjoint, adjoint)
