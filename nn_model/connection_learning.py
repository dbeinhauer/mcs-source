import torch
import torch.nn as nn

class NeuralConnectionMatrix(nn.Module):
    """
    Connection learning module which tries to model local and sparse neural connections.
    """
    def __init__(self, n_neurons: int, n_features: int = 3, ):
        super().__init__()
        assert n_neurons > 0
        assert n_features > 0

        self.n_neurons = n_neurons
        self.n_features = n_features
        # create (n_features) trainable parameters for each neuron. We're hoping that the features
        # will eventually converge to meaningful values that correlate with neural attributes like
        # orientation preference, position, phase...
        self.feature_vectors = nn.Parameter(torch.Tensor(n_neurons, n_features))  # shape: (N, 3)
        nn.init.kaiming_normal_(self.feature_vectors, nonlinearity='relu')
        # DNN that takes in neuron_i, neuron_j and infers the weight between them based on their (trained) features.
        # This is computationally expensive, but it greatly reduces parameter count compared to traditional nn.Linear(n,n).
        self.weight_dnn = nn.Sequential(
            nn.Linear(n_features * 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self) -> torch.Tensor:
        # all possible pairs of neural features
        expanded = self.feature_vectors.expand(self.n_neurons, self.n_neurons, self.n_features)
        x_left = expanded.reshape(self.n_neurons ** 2, self.n_features)
        x_right = expanded.transpose(0, 1).reshape(self.n_neurons ** 2, self.n_features)
        x = torch.cat([x_left, x_right], dim=1)
        # infer connection weight based on learned neural features
        weights = self.weight_dnn(x) # shape (N^2)
        # transform to shape (N, N)
        return weights.view(self.n_neurons, self.n_neurons)

class SparseAffineTransform(nn.Module):
    """
    Calculates X * W^T + b, where W is calculated by the connection learning module.
    """
    def __init__(self, n_neurons: int, n_features: int = 3):
        super().__init__()
        self.weight = NeuralConnectionMatrix(n_neurons, n_features)
        self.bias = nn.Parameter(torch.zeros(n_neurons))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w = self.weight()
        return input @ w.t() + self.bias