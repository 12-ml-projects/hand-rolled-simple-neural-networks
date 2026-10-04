"""One entry per operator, mapping ours onto ONNX's.

An entry is handed the node, its already-resolved input names and its own
output name, and returns the ONNX nodes that implement it.
"""

from typing import Callable, Type, TypeAlias, cast

import numpy as np

from onnx import NodeProto
from onnx.helper import make_node
from onnx.numpy_helper import from_array
from src.dag import DAG
from src.operators import (
    Abs,
    Add,
    MatMul,
    Mul,
    Neg,
    Operator,
    Pow,
    ReLU,
    Sub,
    Sum,
    TrueDiv,
    UnaryPow,
)

Emit: TypeAlias = Callable[[DAG, list[str], str], list[NodeProto]]


def simple(op_type: str) -> Emit:
    """An operator with no state: one node, same operands, no attributes."""

    def emit(node: DAG, inputs: list[str], output: str) -> list[NodeProto]:
        return [make_node(op_type, inputs=inputs, outputs=[output])]

    return emit


def unary_pow(node: DAG, inputs: list[str], output: str) -> list[NodeProto]:
    """x ** n for an n that is operator state rather than a graph node.

    ONNX keeps no values outside tensors, so the exponent has to re-enter the
    graph as a Constant feeding an ordinary two-input Pow. Its name is derived
    from the output's, which is already unique.
    """
    operator = cast(UnaryPow, node.operator)
    exp_name = f"{output}_exponent"

    return [
        make_node(
            "Constant",
            inputs=[],
            outputs=[exp_name],
            value=from_array(operator.exponent, name=exp_name),
        ),
        make_node("Pow", inputs=[*inputs, exp_name], outputs=[output]),
    ]


def reduce_sum(node: DAG, inputs: list[str], output: str) -> list[NodeProto]:
    """Sum, whose axes are metadata about traversal rather than a value.

    ONNX agrees about keepdims, which is an attribute, and disagrees about the
    axes, which became an input at opset 13 -- so naming axes costs a Constant.
    Leaving that input off means every axis, which is what axis=None asks for.
    """
    operator = cast(Sum, node.operator)
    keepdims = int(operator.keepdims)

    if operator.axis is None:
        return [
            make_node("ReduceSum", inputs=inputs, outputs=[output], keepdims=keepdims)
        ]

    axes = f"{output}_axes"

    return [
        make_node(
            "Constant",
            inputs=[],
            outputs=[axes],
            value=from_array(
                np.atleast_1d(np.asarray(operator.axis, dtype=np.int64)), axes
            ),
        ),
        make_node(
            "ReduceSum",
            inputs=[*inputs, axes],
            outputs=[output],
            keepdims=keepdims,
        ),
    ]


OPERATOR_MAP: dict[Type[Operator], Emit] = {
    Abs: simple("Abs"),
    Add: simple("Add"),
    MatMul: simple("MatMul"),
    Mul: simple("Mul"),
    Neg: simple("Neg"),
    Pow: simple("Pow"),
    ReLU: simple("Relu"),
    Sub: simple("Sub"),
    Sum: reduce_sum,
    TrueDiv: simple("Div"),
    UnaryPow: unary_pow,
}
