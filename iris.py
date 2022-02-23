from mlp import MultiLayerPerceptron
from layer import InputLayer, OutputLayer
from action_func import ActionFunction

if __name__ == '__main__':
    mlp = MultiLayerPerceptron(threshold=0.1, stepmax=1000, repetition=5, acceptable_error=0.05)
    mlp.load_data('iris.csv')
    mlp.max_minx_normalization_data()
    mlp.data_separation(0.8, 0.2)
    mlp.add_layer(InputLayer(4))
    mlp.add_hidden_layer_hecht_nielsen()
    mlp.add_layer(OutputLayer(3, ActionFunction.bipolar_sigmoid, ActionFunction.bipolar_derivative_sigmoid))
    mlp.train(min=-0.5, max=0.5)
