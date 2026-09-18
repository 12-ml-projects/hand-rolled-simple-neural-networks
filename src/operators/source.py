from .operator import Operator


class Source(Operator):
    def forward(self, x):
        return x
