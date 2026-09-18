from .operator import Operator


class TrueDiv(Operator):
    def forward(self, x, y):
        return x / y
