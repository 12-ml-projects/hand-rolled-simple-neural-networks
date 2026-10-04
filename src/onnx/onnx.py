from typing import Sequence

from onnx import ModelProto, NodeProto, TensorProto, ValueInfoProto
from onnx.checker import check_model
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_tensor_value_info,
)
from onnx.numpy_helper import from_array
from src.dag import DAG
from src.onnx.operator_map import OPERATOR_MAP
from src.tensor import Tensor

OPSET = 17
IR_VERSION = 8


def make_onnx_model(
    final_node: Tensor,
    inputs: Sequence[Tensor],
    parameters: Sequence[Tensor] = (),
    model_name: str = "model",
) -> ModelProto:
    """Trace a graph into an ONNX model.

    Every leaf is one of three things. A declared input becomes a graph input,
    carrying no data. A parameter becomes an initializer, which is an input whose
    value travels inside the file. Anything left over is a constant someone wrote
    into an expression, and becomes a Constant node.
    """
    dag = final_node._dag
    order = list(dag._topological_order())
    node_names: dict[DAG, str] = {node: str(idx) for idx, node in enumerate(order)}

    input_nodes = {tensor._dag for tensor in inputs}
    parameter_nodes = {tensor._dag for tensor in parameters}

    onnx_inputs: list[ValueInfoProto] = [
        make_tensor_value_info(
            node_names[tensor._dag], TensorProto.DOUBLE, tensor.value.shape
        )
        for tensor in inputs
    ]
    onnx_initializers: list[TensorProto] = []
    onnx_nodes: list[NodeProto] = []
    onnx_outputs: list[ValueInfoProto] = [
        make_tensor_value_info(
            node_names[dag], TensorProto.DOUBLE, final_node.value.shape
        )
    ]

    for node in order:
        if node in input_nodes:
            continue

        if node.is_leaf():
            name = node_names[node]

            if node in parameter_nodes:
                onnx_initializers.append(from_array(node.value, name))
            else:
                onnx_nodes.append(
                    make_node(
                        "Constant",
                        inputs=[],
                        outputs=[name],
                        value=from_array(node.value, name),
                    )
                )
            continue

        emit = OPERATOR_MAP[type(node.operator)]
        onnx_nodes.extend(
            emit(
                node,
                [node_names[dep] for dep in node.dependencies],
                node_names[node],
            )
        )

    graph = make_graph(
        nodes=onnx_nodes,
        name=model_name,
        inputs=onnx_inputs,
        outputs=onnx_outputs,
        initializer=onnx_initializers,
    )

    model = make_model(graph, opset_imports=[make_opsetid("", OPSET)])
    # A second, independent version stamp, which make_model() also takes from
    # the installed onnx -- 1.23 writes IR 14, which onnxruntime refuses. Pinned
    # to the IR version opset 17 originally shipped with. check_model() does not
    # catch this; only a runtime does.
    model.ir_version = IR_VERSION

    check_model(model)

    return model
