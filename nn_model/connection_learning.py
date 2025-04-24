from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from nn_model.globals import POS_ORI_DICT, INHIBITORY_LAYERS, MODEL_SIZES, LAYER_TO_PARENT


def get_features(layer_name_pre, layer_name_post):
    return {
        key: (value.clone(), POS_ORI_DICT[layer_name_post][key].clone())
        for key, value in POS_ORI_DICT[layer_name_pre].items()
    }


class Autapse(nn.Module):
    """
    Learnable self‑connection weights for inhibitory neurons.
    Autapses are self-to-self connections, most common in inhibitory neurons of the visual cortex.
    """

    def __init__(self, n_neurons: int, sharp: bool = False, initial_mean=-8, initial_std=1):
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
    :return: tensor with mean 0 and std 1, same shape
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


def encode_orientation(x: Tensor, y: Tensor) -> Tensor:
    """
    Maps orientation tensor to range [0, 1] using cosine transform.
    :param x: pre-synaptic orientation tensor, shape (m,)
    :param y: post-synaptic orientation tensor, shape (n,)
    :return: encoded pair-wise orientation tensor, shape (m * n,)
    """
    delta = pairwise_delta(x, y).remainder(torch.pi)
    delta = delta - (torch.pi / 2)
    return torch.cos(delta) ** 2


def encode_phase(x: Tensor, y) -> Tensor:
    """
    Maps phase tensor to range [0, 1] using cosine transform.
    :param x: pre-synaptic phase tensor, shape (m,)
    :param y: post-synaptic phase tensor, shape (n,)
    :return: encoded pair-wise phase tensor, shape (m * n,)
    """
    delta = pairwise_delta(x, y).remainder(torch.pi * 2)
    delta = delta - torch.pi
    return torch.cos(delta / 2) ** 2


class NeuralConnectionGenerator(nn.Module):
    """
    Connection learning module which constructs a connection matrix W
    based on local features (location, orientation, phase) of the neurons.

    Can be used for modeling of lateral connections within the same biological layer, specifically:

    E->E, I->I, E->I, I->E connections within L2/3 and L4 layers.

    Phase is only used for L4 layers.

    In the case of E->E / I->I connections, autapses (self-to-self connections) are handled separately.
    """

    def __init__(self, layer_name_pre: str, layer_name_post: str):
        """
        :param layer_name_pre: Name of the layer with pre-synaptic neurons.
        :param layer_name_post: Name of the layer with post-synaptic neurons.
        """
        super().__init__()
        assert LAYER_TO_PARENT[layer_name_pre] == LAYER_TO_PARENT[layer_name_post], 'Neurons are expected to some from the same biological layer.'
        self.in_features = MODEL_SIZES[layer_name_pre]
        self.out_features = MODEL_SIZES[layer_name_post]
        self.polarity = -1 if layer_name_pre in INHIBITORY_LAYERS else 1
        # set flags
        self.is_inh = self.polarity < 0
        self.has_phase = 'L4' in layer_name_pre.upper()
        self.has_self_connection = layer_name_pre == layer_name_post
        # prepare features
        x = self.encode_neural_features(layer_name_pre, layer_name_post)
        self.register_buffer('features', x)
        self.hidden_size = 16
        # handle self-to-self-connections
        if self.is_inh and self.has_self_connection:
            self.autapse = Autapse(self.in_features)
        # initialize DNN
        self.weight_dnn = nn.Sequential(
            nn.Linear(self.features.shape[1], self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, 1),
        )

    def encode_neural_features(self, layer_name_pre: str, layer_name_post: str):
        """
        Creates a tensor representing the neural features.
        :param layer_name_pre: Name of the layer with pre-synaptic neurons.
        :param layer_name_post: Name of the layer with post-synaptic neurons.
        :return: Encoded neural features, shape (n_neurons^2,k) where k is the number of features.
        """
        features = get_features(layer_name_pre, layer_name_post)
        # relative position
        x = features['x']
        y = features['y']
        ori = features['ori']
        dx = pairwise_delta(*x)
        dy = pairwise_delta(*y)
        # relative distance
        dist = (dx ** 2 + dy ** 2) ** 0.5
        dist = dist / 5.
        # relative orientation
        ori = encode_orientation(*ori)
        dx = standard_scale(dx)
        dy = standard_scale(dy)
        res = torch.column_stack((dist, dx, dy, ori,))
        # relative phase
        if self.has_phase:
            phase = features['phase'] if self.has_phase else None
            phase = encode_phase(*phase)
            res = torch.column_stack((res, phase))
        return res

    def forward(self) -> torch.Tensor:
        weights = self.weight_dnn(self.features)  # shape (N^2)
        weights = self.polarity * nn.functional.softplus(self.polarity * weights)
        weights = weights.view(self.out_features, self.in_features)
        if not self.has_self_connection:
            return weights
        if self.is_inh:
            weights.diagonal().copy_(self.autapse())
        else:
            weights.fill_diagonal_(0)
        return weights


class ConnectionAffine(nn.Module):
    """
    Calculates X * W^T + b, where W is calculated by the connection learning module.
    """

    def __init__(self, layer_name_pre: str, layer_name_post: str):
        super().__init__()
        self.weight = NeuralConnectionGenerator(layer_name_pre, layer_name_post)
        self.bias = nn.Parameter(torch.zeros(MODEL_SIZES[layer_name_post], ))

    def forward(self, input: Tensor) -> Tensor:
        return nn.functional.linear(input, self.weight(), self.bias)
