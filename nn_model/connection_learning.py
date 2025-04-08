import torch
import torch.nn as nn
from torch import Tensor
from nn_model.globals import POS_ORI_DICT, INHIBITORY_LAYERS, DEVICE


def standard_scale(x: Tensor) -> Tensor:
    """
    Standard scaling function for pytorch tensors.
    Credit: https://discuss.pytorch.org/t/pytorch-tensor-scaling/38576/2
    :param x: input tensor
    :return: tensor with mean 0 and std 1
    """
    mean = x.mean(0, keepdim=True)
    std = x.std(0, keepdim=True)
    return (x - mean) / std


def pairwise_delta(x: Tensor) -> Tensor:
    """
    Computes pairwise delta between two tensors.
    :param x: input tensor, size n
    :return: pairwise delta, size n^2
    """
    return (x.unsqueeze(0) - x.unsqueeze(1)).flatten()


def encode_orientation(x: Tensor) -> Tensor:
    """
    Encodes orientation tensor x so that it can be passed into a dnn.
    :param x: orientation tensor
    :return: encoded orientation tensor
    """
    delta = pairwise_delta(x).remainder(torch.pi)
    return 0.5 * (1 + torch.cos(2 * delta))  # same as torch.cos(delta) ** 2


def encode_phase(x: Tensor) -> Tensor:
    """
    Encodes phase tensor x so that it can be passed into a dnn.
    :param x: phase tensor
    :return: encoded phase tensor
    """
    delta = pairwise_delta(x).remainder(torch.pi * 2)
    return 0.5 * (1 + torch.cos(delta))  # same as torch.cos(x/2) ** 2


class NeuralConnectionMatrix(nn.Module):
    """
    Connection learning module which tries to model local and sparse neural connections.
    """

    def __init__(self, n_neurons: int, layer_name: str, n_features: int = 3, ):
        super().__init__()
        assert n_neurons > 0
        assert n_features > 0

        self.n_neurons = n_neurons
        self.n_features = n_features
        self.alpha = -1 if layer_name in INHIBITORY_LAYERS else 1
        self.is_l4 = 'L4' in layer_name.upper()
        self.x = self.encode_neural_features(layer_name)

        in_features = self.x.shape[1]

        self.hidden_size = 16

        self.weight_dnn = nn.Sequential(
            nn.Linear(in_features, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, 1),
        )

    def encode_neural_features(self, layer_name):
        """
        Creates a tensor representing the neural features.
        :param layer_name: name of layer which this module represents.
        :return: encoded neural features, length num_neurons^2
        """
        xyo = POS_ORI_DICT[layer_name]
        x = torch.from_numpy(xyo['x']).float().to(DEVICE).flatten()
        y = torch.from_numpy(xyo['y']).float().to(DEVICE).flatten()
        x = pairwise_delta(x)
        y = pairwise_delta(y)
        # distance
        dist = (x ** 2 + y ** 2) ** 0.5
        dist = dist / dist.max()
        # orientation
        ori = torch.tensor(xyo['ori'], dtype=torch.float).to(DEVICE)
        ori = encode_orientation(ori)
        res = torch.column_stack((dist, standard_scale(x), standard_scale(y), ori,))
        # phase
        if self.is_l4:
            phase = torch.from_numpy(xyo['phase']).float().to(DEVICE)
            phase = encode_phase(phase)
            res = torch.column_stack((res, phase))
        return res

    def forward(self) -> torch.Tensor:
        weights = self.weight_dnn(self.x)  # shape (N^2)
        weights = self.alpha * nn.functional.softplus(self.alpha * weights)
        return weights.view(self.n_neurons, self.n_neurons)


class SparseAffineTransform(nn.Module):
    """
    Calculates X * W^T + b, where W is calculated by the connection learning module.
    """

    def __init__(self, n_neurons: int, layer_name: str, n_features: int = 3):
        super().__init__()
        self.alpha = -1 if layer_name in INHIBITORY_LAYERS else 1
        self.weight = NeuralConnectionMatrix(n_neurons, layer_name, n_features)
        self.bias = nn.Parameter(torch.zeros(n_neurons))

    def forward(self, input: Tensor) -> Tensor:
        return nn.functional.linear(input, self.weight(), self.bias)
