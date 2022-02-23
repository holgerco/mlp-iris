from typing import List, Union
import pandas as pd
import numpy as np
from pandas import Series
from pandas._typing import FilePath
from layer import HiddenLayer, OutputLayer, InputLayer
from learning_rate import LearningRate
from neural_network import NeuralNetwork


class MultiLayerPerceptron(NeuralNetwork):

    def __init__(self, stepmax, *args, **kwargs):
        super(MultiLayerPerceptron, self).__init__()
        self.layers: List[Union[HiddenLayer, OutputLayer, InputLayer]] = list()
        self.delta_weights = []
        self.epoch = 0
        self.stepmax = stepmax
        self.threshold = kwargs.get('threshold', 0.1)
        self.mu = kwargs.get('mu', 0.5)
        self.learning_rate = kwargs.get('learning_rate', LearningRate.DEFAULT)
        self.repetition = kwargs.get('repetition', 1)
        self.acceptable_error = kwargs.get('acceptable_error', 0.1)

    def start(self):
        super(MultiLayerPerceptron, self).start()
        self.epoch = 0

    @property
    def accuracy(self):
        return 1 - self.error_rate

    def restart(self):
        self.epoch = 0
        super(MultiLayerPerceptron, self).restart()

    def load_data(self, path: FilePath) -> None:
        data = pd.read_csv(path)  # load file
        self.raw_data = pd.DataFrame(data)  # create data structure

    def max_minx_normalization_data(self):
        # apply normalization techniques
        for column in self.raw_data.columns:
            if 'input' in column:
                self.raw_data[column] = (self.raw_data[column] - self.raw_data[column].min()) / (
                        self.raw_data[column].max() - self.raw_data[column].min())

    def data_separation(self, train_weights, test_weights):
        if train_weights + test_weights == 1:
            train_data = self.raw_data.sample(frac=train_weights)
            test_data = self.raw_data.drop(train_data.index)
        else:
            train_data = self.raw_data.sample(frac=train_weights)
            test_data = self.raw_data.sample(frac=test_weights)

        self.train_inputs = train_data.filter(regex='input')  # train inputs separation
        self.train_targets = train_data.filter(regex='target')  # train targets separation
        self.test_inputs = test_data.filter(regex='input')  # test inputs separation
        self.test_targets = test_data.filter(regex='target')  # test targets separation

    def add_layer(self, layer: Union[InputLayer, HiddenLayer, OutputLayer]):
        self.layers.append(layer)

    def add_hidden_layer_hecht_nielsen(self):
        number_of_hidden_cell = self._hecht_nielsen_cell_count()
        hidden_layer = HiddenLayer(
            cells_count=number_of_hidden_cell,
            func=self.action_functions.bipolar_sigmoid,
            derivative_func=self.action_functions.bipolar_derivative_sigmoid,
        )
        self.add_layer(hidden_layer)

    def _hecht_nielsen_cell_count(self):
        """
        calculate the number of hidden layer cells based on the hecht-nielsen rule
        :return: cell count of hidden layer
        """
        input_layer = self.layers[0]
        return 2 * (len(input_layer.cells) + 1) + 1

    def define_zero_weights_matrix(self):
        for index, layer in enumerate(self.layers):
            if type(layer) in [HiddenLayer, OutputLayer]:
                previous_layer = self.layers[index - 1]
                input_layer_cells_count = len(previous_layer.cells)
                output_layer_cells_count = len(layer.cells)
                zero_matrix = np.zeros(shape=[input_layer_cells_count + 1, output_layer_cells_count])
                weights = pd.DataFrame(zero_matrix)
                self.weights.append(weights)
                delta_weights = pd.DataFrame(zero_matrix)
                self.delta_weights.append(delta_weights)
            else:
                continue

    def define_weights_matrix(self, min=-0.5, max=0.5):
        self.weights.clear()
        self.delta_weights.clear()
        for previous_index, layer in enumerate(self.layers[1:]):
            previous_layer = self.layers[previous_index]
            input_layer_cells_count = len(previous_layer.cells)
            output_layer_cells_count = len(layer.cells)
            matrix = np.random.uniform(
                min,
                max,
                size=(input_layer_cells_count + 1, output_layer_cells_count)
            )
            weights = pd.DataFrame(matrix)
            self.weights.append(weights)
            zero_matrix = np.zeros(shape=[input_layer_cells_count + 1, output_layer_cells_count])
            delta_weights = pd.DataFrame(zero_matrix)
            self.delta_weights.append(delta_weights)

    def nguyen_widrow_normalization_weight(self):
        for index, layer in enumerate(self.layers[:-1]):
            next_layer = self.layers[index + 1]
            n = len(layer.cells)
            p = len(next_layer.cells)
            beta = 0.7 * (p ** (1 / n))
            weights: pd.DataFrame = self.weights[index]
            for column in weights.columns:
                _v_old = weights[column].values
                _v_old_norm = np.linalg.norm(_v_old)
                for i, x_i in enumerate([layer.bias, *layer.cells]):
                    weights[column][i] = beta * weights[column][i] / _v_old_norm
            self.weights[index] = weights

    def train(self, min=-0.5, max=0.5):
        for index in range(self.repetition):
            rep = index + 1
            self.define_weights_matrix(min=min, max=max)
            self.nguyen_widrow_normalization_weight()
            self.learn()
            self.test()
            self.result(rep)

    def learn(self):
        self.start()
        while self.stopping_condition_have_not_been_raised:
            self.info()
            for index, row in self.train_inputs.iterrows():
                self.feedforward(row)
                self.back_propagate(self.train_targets.loc[[index]])
                self.weight_adjustment()
            self.check_stopping_condition()

    def info(self):
        self.epoch += 1
        print(f'epoch:{self.epoch} error:{self.error_rate:.4f} ...', end='\r')

    def feedforward(self, _input: Series):
        print(f'epoch {self.epoch} error:{self.error_rate:.4f} feeding forward...\t\t   |', end='\r')
        signal = _input.values
        self.layers[0].cells = signal
        for index, layer in enumerate(self.layers[:-1]):
            layer: Union[InputLayer, HiddenLayer]
            weights = self.weights[index]
            next_layer = self.layers[index + 1]
            next_layer: Union[HiddenLayer, OutputLayer]
            signal = []
            in_signal = []
            for j, y_j in enumerate(next_layer.cells):
                z_in_j = 0
                for i, x_i in enumerate([layer.bias, *layer.cells]):
                    z_in_j += x_i * weights[j][i]
                z_j = next_layer.func(z_in_j)
                in_signal.append(z_in_j)
                signal.append(z_j)
            next_layer.cells = signal
            next_layer.in_cells = in_signal

    def back_propagate(self, expected: Series):
        print(f'epoch {self.epoch} error:{self.error_rate:.4f} back propagation...\t\t', end='\r')
        delta = self.output_layer_back_propagate(expected)
        self.hidden_layer_back_propagate(delta)

    def output_layer_back_propagate(self, expected: Series) -> List[float]:
        target = expected.values
        output_layer = self.layers[-1]
        previous_layer = self.layers[-2]
        delta = []
        for j, y_j in enumerate(output_layer.cells):
            delta_j = (target[0][j] - y_j) * output_layer.derivative_func(output_layer.in_cells[j])
            delta.append(delta_j)
            for i, x_i in enumerate([previous_layer.bias, *previous_layer.cells]):
                delta_w_i_j = self.threshold * delta_j * x_i
                self.delta_weights[-1][j][i] = self.calculate_learning_delta(
                    old_delta=self.delta_weights[-1][j][i],
                    new_delta=delta_w_i_j,
                )
        return delta

    def hidden_layer_back_propagate(self, delta: List[float]):
        next_delta = []
        for index, layer in enumerate(self.layers[-2:0:-1]):
            real_index = len(self.layers) - (index + 1) - 1
            previous_layer = self.layers[real_index - 1]
            for j, y_j in enumerate(layer.cells):
                delta_in_j = 0
                for k, w_j_k in enumerate(self.weights[real_index].values[j]):
                    delta_in_j += delta[k] * w_j_k
                delta_j = delta_in_j * layer.derivative_func(layer.in_cells[j])
                next_delta.append(delta_j)
                for i, x_i in enumerate([previous_layer.bias, *previous_layer.cells]):
                    delta_w_i_j = self.threshold * delta_j * x_i
                    self.delta_weights[real_index - 1][j][i] = self.calculate_learning_delta(
                        old_delta=self.delta_weights[real_index - 1][j][i],
                        new_delta=delta_w_i_j,
                    )
            delta = next_delta
            next_delta = []

    def calculate_learning_delta(self, old_delta, new_delta) -> float:
        _delta = 0
        if self.learning_rate == LearningRate.DEFAULT:
            _delta = new_delta
        elif self.learning_rate == LearningRate.MOMENTUM:
            _delta = old_delta * self.mu + new_delta
        return _delta

    def weight_adjustment(self):
        print(f'epoch {self.epoch} error:{self.error_rate:.4f} weight adjustment...\t\t', end='\r')
        for index, weight in enumerate(self.weights):
            weight: pd.DataFrame
            for i, row in enumerate(weight.values):
                for j, column in enumerate(row):
                    self.weights[index][j][i] += self.delta_weights[index][j][i]

    def check_stopping_condition(self):
        print(f'epoch {self.epoch} error:{self.error_rate:.4f} checking... \t\t', end='\r')
        self.calculate_accuracy()
        self.check_epoch()

    def calculate_accuracy(self):
        corrects = self.test()
        error_rate = 1 - (corrects / len(self.test_inputs.values))
        if error_rate < self.acceptable_error:
            self.stop()
        self.error_rate = error_rate
        self.error_rate_series.append(error_rate)

    def check_epoch(self):
        if self.epoch == self.stepmax:
            self.stop()

    def test(self):
        corrects = 0
        for index, row in self.test_inputs.iterrows():
            self.calculate_result(row)
            expected = self.test_targets.loc[[index]]
            if self.diagnosis(expected):
                corrects += 1
        return corrects

    def result(self, repetition: int):
        hidden = ','.join([str(len(layer.cells)) for layer in self.layers[1:-1]])
        print(
            F'Repetition {repetition} -> hidden:{hidden} threshold:{self.threshold} '
            F'epoch:{self.epoch} acceptable error:{self.acceptable_error} '
            F'error:{self.error_rate:.2f} accuracy:{self.accuracy:.2f}'
        )

    def calculate_result(self, _input: Series):
        signal = _input.values
        self.layers[0].cells = signal
        for index, layer in enumerate(self.layers[:-1]):
            layer: Union[InputLayer, HiddenLayer]
            weights = self.weights[index]
            next_layer = self.layers[index + 1]
            next_layer: Union[HiddenLayer, OutputLayer]
            signal = []
            for j, y_j in enumerate(next_layer.cells):
                z_in_j = 0
                for i, x_i in enumerate([layer.bias, *layer.cells]):
                    z_in_j += x_i * weights[j][i]
                z_j = next_layer.func(z_in_j)
                signal.append(z_j)
            next_layer.cells = signal

    def diagnosis(self, expected: Series):
        target = expected.values[0]
        output_layer = self.layers[-1]
        result = output_layer.cells
        return self.is_correct(target, result)

    @staticmethod
    def is_correct(target, result):
        prediction = -1
        for index, cell in enumerate(result):
            if prediction == -1:
                prediction = index
            elif result[prediction] < cell:
                prediction = index
        if target[prediction] == 1:
            return True
        return False

    @property
    def stopping_condition_have_not_been_raised(self):
        return self.is_running()
