from .operator import Operator


class Neg(Operator):
    def forward(self, x):
        return -x
