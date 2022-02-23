from abc import ABC
from action_func import ActionFunction
from layer import Layer

class NeuralNetwork(ABC):
    action_functions = ActionFunction()

    class State:
        ready = 'READY'
        running = 'RUNNING'
        stop = 'STOP'

    def __init__(self):
        self.raw_data = None
        self.train_inputs = None
        self.train_targets = None
        self.test_inputs = None
        self.test_targets = None
        self.weights = []
        self.state = NeuralNetwork.State.ready
        self.error_rate = None
        self.error_rate_series = None

    def is_ready(self):
        return self.state == NeuralNetwork.State.ready

    def is_running(self):
        return self.state == NeuralNetwork.State.running

    def is_stopped(self):
        return self.state == NeuralNetwork.State.stop

    def start(self):
        self.state = NeuralNetwork.State.running
        self.error_rate = 1
        self.error_rate_series = []

    def stop(self):
        self.state = NeuralNetwork.State.stop

    def restart(self):
        self.state = NeuralNetwork.State.ready
        self.error_rate = 1
        self.error_rate_series = []
