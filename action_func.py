import numpy as np


class ActionFunction:

    @staticmethod
    def bipolar_sigmoid(x):
        return (1 - np.exp(-x)) / (1 + np.exp(-x))

    @staticmethod
    def bipolar_derivative_sigmoid(x):
        return (1 + ActionFunction.bipolar_sigmoid(x)) * (1 - ActionFunction.bipolar_sigmoid(x)) / 2

    @staticmethod
    def logistics(x):
        if x > 0:
            return np.log(1 + x)
        elif x < 0:
            return -np.log(1 - x)
        return 0

    @staticmethod
    def logistics_derivative(x):
        if x > 0:
            return 1 / (1 + x)
        elif x < 0:
            return 1 - x
        return 1

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_derivative(x):
        return 1
