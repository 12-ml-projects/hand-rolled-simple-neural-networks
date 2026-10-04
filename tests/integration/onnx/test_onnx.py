from typing import Any, Type

import numpy as np
import pytest
from onnxruntime import InferenceSession

from onnx.numpy_helper import to_array
from src.ffnn.ffnn import Mlp
from src.onnx.onnx import make_onnx_model
from src.operators import Abs, Add, MatMul, Mul, Neg, Operator, Pow, Sub, TrueDiv
from src.operators.pow import UnaryPow
from src.operators.relu import ReLU
from src.operators.sum import Axis
from src.tensor import Tensor


def unary_op_model(operand: Tensor, operator: Type[Operator], **kwargs: Any) -> Tensor:
    return operand._apply(operator(**kwargs), operand)


class TestONNX:
    @pytest.mark.parametrize(
        "operand,operator",
        [
            (Tensor([1.0, 2.0, 3.0]), Abs),
            (Tensor([4.0, 5.0, 6.0]), Neg),
            (Tensor([7.0, 8.0, 9.0]), ReLU),
        ],
    )
    def test_unary_ops_onnx_models(
        self, operand: Tensor, operator: Type[Operator]
    ) -> None:
        output = unary_op_model(operand, operator)

        model = make_onnx_model(final_node=output, inputs=[operand])

        graph = model.graph

        assert len(graph.input) == 1
        assert len(graph.output) == 1
        assert len(graph.node) == 1

        # Names are the operand's position in the topological order
        assert list(graph.node[0].input) == ["0"]
        assert list(graph.node[0].output) == ["1"]

        assert graph.input[0].name == "0"
        assert graph.output[0].name == "1"

    @pytest.mark.parametrize(
        "operand_1, operand_2, operator",
        [
            (Tensor([4.0, 5.0, 6.0]), Tensor([1.0, 2.0, 3.0]), Sub),
            (Tensor([4.0, 5.0, 6.0]), Tensor([1.0, 2.0, 3.0]), Add),
            (Tensor([4.0, 5.0, 6.0]), Tensor([1.0, 2.0, 3.0]), Mul),
            (Tensor([[4.0, 5.0, 6.0]]), Tensor([[1.0], [2.0], [3.0]]), MatMul),
            (Tensor([4.0, 5.0, 6.0]), Tensor([1.0, 2.0, 3.0]), Pow),
            (Tensor([4.0, 5.0, 6.0]), Tensor([1.0, 2.0, 3.0]), TrueDiv),
        ],
    )
    def test_binary_ops_onnx_models(
        self, operand_1: Tensor, operand_2: Tensor, operator: Type[Operator]
    ) -> None:
        output = operand_1._apply(operator(), operand_1, operand_2)

        model = make_onnx_model(final_node=output, inputs=[operand_1, operand_2])

        graph = model.graph

        assert len(graph.input) == 2
        assert len(graph.output) == 1
        assert len(graph.node) == 1

        assert list(graph.node[0].input) == ["0", "1"]
        assert list(graph.node[0].output) == ["2"]

        assert graph.input[0].name == "0"
        assert graph.input[1].name == "1"
        assert graph.output[0].name == "2"

    def test_unary_pow_emits_a_constant_and_a_pow(self) -> None:
        operand = Tensor([4.0, 5.0, 6.0])
        output = unary_op_model(operand, UnaryPow, exponent=np.array(2.0))

        graph = make_onnx_model(final_node=output, inputs=[operand]).graph

        constant, power = graph.node

        assert constant.op_type == "Constant"
        assert power.op_type == "Pow"

        assert list(constant.output) == ["1_exponent"]
        assert list(power.input) == ["0", "1_exponent"]
        assert list(power.output) == ["1"]

        (value,) = constant.attribute
        assert to_array(value.t) == 2.0

    @pytest.mark.parametrize("keepdims", [True, False])
    @pytest.mark.parametrize("axis", [None, 0, 1, (0, 1)])
    def test_sum_names_its_axes(self, axis: Axis, keepdims: bool) -> None:
        # ONNX made axes an input at opset 13, so naming any costs a Constant.
        operand = Tensor([[1.0, 2.0], [3.0, 4.0]])
        output = operand.sum(axis, keepdims)

        graph = make_onnx_model(final_node=output, inputs=[operand]).graph

        *constants, reduce_sum = graph.node

        assert reduce_sum.op_type == "ReduceSum"
        assert {"keepdims": keepdims} == {
            attribute.name: bool(attribute.i) for attribute in reduce_sum.attribute
        }

        if axis is None:
            assert constants == []
            assert reduce_sum.input == ["0"]

            return

        (axes,) = constants
        assert reduce_sum.input == ["0", "1_axes"]
        assert to_array(axes.attribute[0].t).tolist() == np.atleast_1d(axis).tolist()


class TestAgainstOnnxRuntime:
    """A graph is only known to be right if something else agrees about it.

    check_model validates structure, not meaning: it passes a graph whose
    MatMul operands are transposed. onnxruntime is a separate implementation of
    ONNX's semantics, so it cannot share a mistake made here.
    """

    def test_a_graph_using_every_mapped_operator(self) -> None:
        rng = np.random.default_rng(0)

        features = rng.normal(size=(4, 3))
        inputs = Tensor(features, requires_grad=False)
        weights = Tensor(rng.normal(size=(3, 5)))
        bias = Tensor(rng.normal(size=5))

        # Deliberately contrived: it exists to reach every entry in the op map
        hidden = (inputs @ weights + bias).relu()
        centred = hidden - hidden.sum(axis=-1, keepdims=True) / 5.0
        output = abs(-centred) ** 2.0 / (1.0 + centred.sum())

        model = make_onnx_model(
            final_node=output, inputs=[inputs], parameters=[weights, bias]
        )

        emitted = {node.op_type for node in model.graph.node}
        assert emitted == {
            "Abs",
            "Add",
            "Constant",
            "Div",
            "MatMul",
            "Neg",
            "Pow",
            "ReduceSum",
            "Relu",
            "Sub",
        }

        assert [info.name for info in model.graph.input] == ["0"]
        assert len(model.graph.initializer) == 2

        session = InferenceSession(model.SerializeToString())
        (theirs,) = session.run(None, {"0": features})
        ours = output.value

        assert theirs == pytest.approx(ours)

    def test_the_network_exports_as_a_runnable_model(self) -> None:
        rng = np.random.default_rng(0)

        mlp = Mlp(input_size=6, hidden_size=4, n_classes=3, rng=rng)
        features = rng.normal(size=(5, 6))

        model = mlp.export(Tensor(features, requires_grad=False))

        (declared,) = model.graph.input
        assert len(model.graph.initializer) == 4

        for parameter, initializer in zip(mlp.parameters(), model.graph.initializer):
            assert to_array(initializer) == pytest.approx(parameter.value)

        session = InferenceSession(model.SerializeToString())
        (theirs,) = session.run(None, {declared.name: features})

        assert theirs == pytest.approx(mlp(Tensor(features)).value)
