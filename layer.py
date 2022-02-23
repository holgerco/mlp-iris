from typing import List
from abc import ABC


class Layer(ABC):
    cells = []
    in_cells = []
    pass


class InputLayer(Layer):
    def __init__(self, cells_count):
        self.cells = [0 for _ in range(cells_count)]
        self.bias = 1

    def train(self, data: List[str]):
        assert len(data) == len(self.cells), F'length of data must be equal to number of cells'


class HiddenLayer(Layer):
    def __init__(self, cells_count, func, derivative_func):
        self.cells = [0 for _ in range(cells_count)]
        self.in_cells = [0 for _ in range(cells_count)]
        self.func = func
        self.derivative_func = derivative_func
        self.bias = 1


class OutputLayer(Layer):
    def __init__(self, cells_count, func, derivative_func):
        self.cells = [0 for _ in range(cells_count)]
        self.in_cells = [0 for _ in range(cells_count)]
        self.func = func
        self.derivative_func = derivative_func
