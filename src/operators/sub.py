from .operator import Operator


class Sub(Operator):
    def forward(self, x, y):
        return x - y
