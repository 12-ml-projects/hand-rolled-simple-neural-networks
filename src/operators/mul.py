from .operator import Operator


class Mul(Operator):
    def forward(self, x, y):
        return x * y
