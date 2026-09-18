from .operator import Operator


class Add(Operator):
    def forward(self, x, y):
        return x + y
