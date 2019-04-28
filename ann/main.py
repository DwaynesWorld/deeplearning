import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


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


class Sigmoid(Operation):
    def __init__(self, z):
        super().__init__([z])

    def compute(self, z_val):
        return 1 / (1 + np.exp(-z_val))


class Placeholder():
    def __init__(self):
        self.output_nodes = []
        _default_graph.placeholders.append(self)


class Variable():
    def __init__(self, initial_value=None):
        self.value = initial_value
        self.output_nodes = []
        _default_graph.variables.append(self)


class Graph():
    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []

    def set_as_default(self):
        global _default_graph
        _default_graph = self


class Session():
    def traverse_postorder(self, operation):
        node_postorder = []

        def recurse(node):
            if isinstance(node, Operation):
                for input_node in node.input_nodes:
                    recurse(input_node)
            node_postorder.append(node)

        recurse(operation)

        return node_postorder

    def run(self, operation, feed_dict={}):
        nodes_postorder = self.traverse_postorder(operation)

        for node in nodes_postorder:
            if type(node) == Placeholder:
                node.output = feed_dict[node]
            elif type(node) == Variable:
                node.output = node.value
            else:
                node.inputs = [
                    input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)

            if type(node.output) == list:
                node.output = np.array(node.output)

        return operation.output


g = Graph()
g.set_as_default()

# Simple Inputs
# a = Variable(10)
# b = Variable(1)
# x = Placeholder()
# y = Multiply(a, x)
# z = Add(y, b)

# Matrix Inputs
# A = Variable([[10, 20], [30, 40]])
# b = Variable([1, 2])
# x = Placeholder()
# y = MatMult(A, x)
# z = Add(y, b)

# sess = Session()
# result = sess.run(operation=z, feed_dict={x: 10})
# print(result)

# Test sigmoid
# sample_z = np.linspace(-10, 10, 100)
# sample_a = sigmoid(sample_z)

# plt.plot(sample_z, sample_a)
# plt.show()


data = make_blobs(n_samples=50, n_features=2, centers=2, random_state=75)
# print(data[0])
# print(data[1])

features = data[0]
labels = data[1]
# plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='coolwarm')

# x = np.linspace(0, 11, 10)
# y = -x + 5
# plt.plot(x, y)
# plt.show()
# print(np.array([1, 1]).dot(np.array([[8], [10]])) - 5)

x = Placeholder()
w = Variable([1, 1])
b = Variable(-5)
z = Add(MatMult(w, x), b)
a = Sigmoid(z)

sess = Session()

result = sess.run(operation=a, feed_dict={x: [8, 10]})
print(result)

result = sess.run(operation=a, feed_dict={x: [2, -10]})
print(result)
