from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from src.tensor import Tensor

if TYPE_CHECKING:
    from onnx import ModelProto


class Model(ABC):
    """Something with weights that turns a tensor into a tensor.

    Two implementations earn the base class: a network written here, and one
    reconstructed from an imported ONNX graph, which has weights and a forward
    pass but none of the structure a hand-written network has.
    """

    @abstractmethod
    def parameters(self) -> list[Tensor]:
        """Every leaf the optimiser may update.

        Also what tells an export which leaves are weights.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, features: Tensor, /) -> Tensor:
        raise NotImplementedError

    def export(self, features: Tensor, name: str = "model") -> "ModelProto":
        """Trace a forward pass and serialise it as ONNX.

        Takes an example input because the graph only exists once something has
        flowed through it -- the same reason torch.onnx.export does.
        """
        from src.onnx.onnx import make_onnx_model

        return make_onnx_model(self(features), [features], self.parameters(), name)
