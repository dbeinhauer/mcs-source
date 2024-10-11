"""
This source code defines multiple variants of complex neurons
that can be used in the model. Neuron here is represented by 
some small model.
"""

import torch.nn as nn


# Shared complexity module
class SharedComplexity(nn.Module):
    """
    TODO:
    Class defining shared complexity of the layer that should represent
    one neuron (more complex neuron than just one operation).
    """

    def __init__(self, hidden_size, complexity_size: int = 64):

        super(SharedComplexity, self).__init__()
        # Small neural network for shared complexity
        self.complex_layer = nn.Sequential(
            nn.Linear(hidden_size, complexity_size),
            nn.ReLU(),
            nn.Linear(complexity_size, hidden_size),
        )

    def forward(self, hidden):
        return self.complex_layer(hidden)
