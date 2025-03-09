"""
This script defines all classes used as weights constraints. These 
are typically used for determination of excitatory/inhibitory layer.
"""

import torch
from torch import nn

from nn_model.type_variants import WeightTypes
import torch.nn.utils.parametrize as P
import torch.nn.functional as F


class ScaledReLU(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * F.relu(x)

    def right_inverse(self, X: torch.Tensor) -> torch.Tensor:
        return X / self.alpha


class ConstraintRegistrar:
    """
    Responsible for registering constraints on weights.
    The constraint type is based on whether the weights are excitatory or inhibitory.
    """

    def __init__(self, layer_type: WeightTypes):
        assert isinstance(layer_type, WeightTypes)
        if layer_type == WeightTypes.EXCITATORY:
            self.parametrization = ScaledReLU(1)
        elif layer_type == WeightTypes.INHIBITORY:
            self.parametrization = ScaledReLU(-1)

    @staticmethod
    def register_linear(linear: nn.Linear, parametrization: nn.Module) -> nn.Module:
        """
        Should be called once upon construction of model, pytorch handles the subsequent constraints.
        :param linear: instance of nn.Linear which should have its 'weight' constrained.
        :param parametrization: nn.Module which takes weights and returns its constrained version.
        :return: instance of constrained nn.Linear.
        """
        assert isinstance(linear, nn.Linear)
        assert isinstance(parametrization, nn.Module)
        return P.register_parametrization(linear, 'weight', parametrization)

    def register(self, module: nn.Module) -> nn.Module:
        """
        Register constraints for module.
        :param module: instance of nn.Module with multiple nn.Linear layers. For example: CustomRNNCell
        :return: instance of constrained nn.Module.
        """
        if hasattr(module, "weights_hh") and isinstance(module.weights_hh, nn.Linear):
            ConstraintRegistrar.register_linear(module.weights_hh, self.parametrization)
        if hasattr(module, "weights_ih_exc") and isinstance(module.weights_ih_exc, nn.Linear):
            ConstraintRegistrar.register_linear(module.weights_ih_exc, ScaledReLU(1))
        if hasattr(module, "weights_ih_inh") and isinstance(module.weights_ih_inh, nn.Linear):
            ConstraintRegistrar.register_linear(module.weights_ih_inh, ScaledReLU(-1))
        return module


class ExcitatoryConstraint(ConstraintRegistrar):
    def __init__(self):
        super().__init__(WeightTypes.EXCITATORY)


class InhibitoryConstraint(ConstraintRegistrar):
    def __init__(self):
        super().__init__(WeightTypes.INHIBITORY)
