from typing import (Type, List)
import numpy as np


class Graph():
    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []

    def set_as_default(self):
        global _default_graph
        _default_graph = self


class Operation():
    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes
        self.output_nodes = []

        for node in input_nodes:
            node.output_nodes.append(self)

        _default_graph.operations.append(self)

    def compute():
        pass


class Add(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x, y):
        self.inputs = [x, y]
        return x + y


class Subtract(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x, y):
        self.inputs = [x, y]
        return x - y


class Multiply(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x, y):
        self.inputs = [x, y]
        return x * y


class MatMult(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x, y):
        self.inputs = [x, y]
        return x.dot(y)


class Placeholder():
    def __init__(self):
        self.output_nodes = []
        _default_graph.placeholders.append(self)


class Variable():
    def __init__(self, initial_value=None):
        self.value = initial_value
        self.output_nodes = []
        _default_graph.variables.append(self)


g = Graph()
g.set_as_default()

a = Variable(10)
b = Variable(1)

x = Placeholder()

y = Multiply(a, x)
z = Add(y, b)
