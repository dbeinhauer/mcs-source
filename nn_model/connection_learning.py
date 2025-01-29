import torch
import torch.nn as nn


class NeuralConnections(nn.Module):
    def __init(self, n_neurons: int, n_features: int = 3, ):
        super().__init__()
        assert n_neurons > 0
        assert n_features > 0

        self.n_neurons = n_neurons
        self.n_features = n_features
        self.feature_vectors = nn.Parameter(torch.Tensor(n_neurons, n_features))  # shape: (N, 3)
        # dnn that takes in neuron_i, neuron_j and infers the weight between them
        self.weight_dnn = nn.Sequential(
            nn.Linear(n_features * 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        ar = torch.arange(n_neurons)
        self.cartesian_indices = torch.cartesian_prod(ar, ar)

    def forward(self):
        feature_pairs = torch.cat(
            [
                self.feature_vectors[self.cartesian_indices[:, 0]],
                self.feature_vectors[self.cartesian_indices[:, 1]]
            ],
            dim=1
        )
        weights = self.weight_dnn(feature_pairs)
        return weights.view(self.n_neurons, self.n_neurons)
