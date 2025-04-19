from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from nn_model.globals import POS_ORI_DICT, INHIBITORY_LAYERS


class Autapse(nn.Module):
    """
    Learnable self‑connection weights for inhibitory neurons.
    Autapses are self-to-self connections, most common in inhibitory neurons of the visual cortex.
    """

    def __init__(self, n_neurons: int, sharp: bool = False, initial_mean=-9, initial_std=1e-2):
        """
        :param n_neurons: number of neurons in the layer.
        :param sharp: whether to use sharp or softplus function.
        :param initial_mean: initial mean of the embedding.
        :param initial_std: initial standard deviation of the embedding.
        """
        super().__init__()
        self.n_neurons = n_neurons
        initial = torch.normal(mean=initial_mean, std=initial_std, size=(self.n_neurons,))
        self.embedding = nn.Parameter(initial)
        self.constraint = nn.functional.relu if sharp else nn.functional.softplus

    def forward(self) -> Tensor:
        """
        Apply the chosen constraint to the embedding and negate it.

        :return: Negative autapse weights, shape (n_neurons,).        """
        return - self.constraint(self.embedding)


def standard_scale(x: Tensor) -> Tensor:
    """
    Standard scaling function for pytorch tensors.
    Credit: https://discuss.pytorch.org/t/pytorch-tensor-scaling/38576/2
    :param x: input tensor
    :return: tensor with mean 0 and std 1
    """
    mean = x.mean(0, keepdim=True)
    std = x.std(0, keepdim=True, unbiased=False)
    return (x - mean) / std


def pairwise_delta(x: Tensor, y: Optional[Tensor] = None) -> Tensor:
    """
    Computes pairwise delta between two tensors.
    :param x: input tensor, size m
    :param y: input tensor, size n. Defaults to x.
    :return: pairwise delta, shape (m * n,)
    """
    y = x if y is None else y
    return (x.unsqueeze(0) - y.unsqueeze(1)).flatten()


def encode_orientation(x: Tensor) -> Tensor:
    """
    Maps orientation tensor to range [0, 1] using cosine transform.
    :param x: orientation tensor
    :return: encoded orientation tensor
    """
    delta = pairwise_delta(x).remainder(torch.pi)
    delta = delta - (torch.pi / 2)
    return torch.cos(delta) ** 2


def encode_phase(x: Tensor) -> Tensor:
    """
    Maps phase tensor to range [0, 1] using cosine transform.
    :param x: phase tensor
    :return: encoded phase tensor
    """
    delta = pairwise_delta(x).remainder(torch.pi * 2)
    delta = delta - torch.pi
    return torch.cos(delta / 2) ** 2


class NeuralConnectionMatrix(nn.Module):
    """
    Connection learning module which constructs a connection matrix W
    based on local features (location, orientation, phase) of the neurons.
    """

    def __init__(self, n_neurons: int, layer_name: str):
        """
        :param n_neurons: number of neurons in the layer.
        :param layer_name: name of the layer.
        """
        super().__init__()
        assert n_neurons > 0

        self.n_neurons = n_neurons
        self.polarity = -1 if layer_name in INHIBITORY_LAYERS else 1
        self.is_inh = self.polarity < 0
        self.is_l4 = 'L4' in layer_name.upper()
        x = self.encode_neural_features(layer_name)
        self.register_buffer('features', x)
        self.hidden_size = 16
        if self.is_inh:
            self.autapse = Autapse(self.n_neurons)

        self.weight_dnn = nn.Sequential(
            nn.Linear(self.features.shape[1], self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, 1),
        )

    def encode_neural_features(self, layer_name):
        """
        Creates a tensor representing the neural features.
        :param layer_name: name of the layer
        :return: encoded neural features, shape (n_neurons^2,)
        """
        layer_dict = POS_ORI_DICT[layer_name]
        x = layer_dict['x'].clone()
        y = layer_dict['y'].clone()
        dx = pairwise_delta(x)
        dy = pairwise_delta(y)
        # distance
        dist = (dx ** 2 + dy ** 2) ** 0.5
        dist = dist / 5.
        # orientation
        ori = layer_dict['ori'].clone()
        ori = encode_orientation(ori)
        dx = standard_scale(dx)
        dy = standard_scale(dy)
        res = torch.column_stack((dist, dx, dy, ori,))
        # phase
        if self.is_l4:
            phase = layer_dict['phase'].clone()
            phase = encode_phase(phase)
            res = torch.column_stack((res, phase))
        return res

    def forward(self) -> torch.Tensor:
        weights = self.weight_dnn(self.features)  # shape (N^2)
        weights = self.polarity * nn.functional.softplus(self.polarity * weights)
        weights = weights.view(self.n_neurons, self.n_neurons)
        if self.is_inh:
            weights.diagonal().copy_(self.autapse())
        else:
            weights.fill_diagonal_(0)
        return weights


class SparseAffineTransform(nn.Module):
    """
    Calculates X * W^T + b, where W is calculated by the connection learning module.
    """

    def __init__(self, n_neurons: int, layer_name: str):
        super().__init__()
        self.polarity = -1 if layer_name in INHIBITORY_LAYERS else 1
        self.weight = NeuralConnectionMatrix(n_neurons, layer_name)
        self.bias = nn.Parameter(torch.zeros(n_neurons))

    def forward(self, input: Tensor) -> Tensor:
        return nn.functional.linear(input, self.weight(), self.bias)
