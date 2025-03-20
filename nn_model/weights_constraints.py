"""
This script defines all classes used as weights constraints. These 
are typically used for determination of excitatory/inhibitory layer.
"""
from abc import ABC, abstractmethod

import torch
from torch import nn

from nn_model.type_variants import WeightTypes, WeightConstraint
from torch.nn.utils.parametrize import register_parametrization
from torch.nn.functional import relu, softplus
from torch import Tensor


class Constraint(nn.Module, ABC):
    """
    This class is a PyTorch Module with methods that are required by the parametrization API.
    When a tensor (parameter of some Module) is parametrized with this Module, PyTorch stores its original values.
    When the constrained (parametrized) tensor is accessed during a forward pass, it's presented in its constrained state.
    The constrained state is computed as ``constraint.forward(tensor.original)``.
    During backprop, ``tensor.original`` gets updated.
    See also: https://pytorch.org/tutorials/intermediate/parametrizations.html
    """

    def __init__(self, constraint_type: WeightTypes):
        """
        Module constructor.
        :param constraint_type: type of constraint (excitatory or inhibitory)
        """
        super().__init__()
        self.constraint_type: WeightTypes = constraint_type
        self.sign: int = 1 if constraint_type == WeightTypes.EXCITATORY else -1 # desired sign of the constrained tensor

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Transforms tensor to desired form.
        :param x: input tensor
        :return: tensor transformed to constrained domain
        """
        raise NotImplementedError

    @abstractmethod
    def right_inverse(self, x: Tensor) -> Tensor:
        """
        Inverse of forward transformation. The method is used upon initialization of parametrized tensor.
        Definition: forward(right_inverse(x)) = x
        :param x: input tensor
        :return: right inverse of forward transformation.
        """
        raise NotImplementedError


class SharpConstraint(Constraint):
    """
    Constraint which uses ReLU.

    Layer Type | Output
    Excitatory | forward(x) >= 0
    Inhibitory | forward(x) <= 0
    """

    def forward(self, x: Tensor) -> Tensor:
        return self.sign * relu(self.sign * x)

    def right_inverse(self, x: Tensor) -> Tensor:
        return x


def inv_softplus(y: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Inverse of softplus.
    We use torch.expm1 for numerical stability. Eps is added so that the computation doesn't fail for y = 0.
    Source: https://github.com/pytorch/pytorch/issues/72759#issuecomment-1236496693
    """
    return y + torch.log(-torch.expm1(-y) + eps)


class SmoothConstraint(Constraint):
    """
    Constraint which uses softplus.
    Unlike ReLU, this variant is smooth and not lossy.

    Layer Type | Output
    Excitatory | forward(x) > 0
    Inhibitory | forward(x) < 0
    """

    def forward(self, x: Tensor) -> Tensor:
        return self.sign * softplus(self.sign * x, 1, 20)

    def right_inverse(self, x: Tensor) -> Tensor:
        return self.sign * inv_softplus(self.sign * x)


class ConstraintRegistrar:
    """
    Responsible for registering constraints on weights.
    The constraint type is based on whether the weights are excitatory or inhibitory.
    """
    CONSTRAINT_MAPPING = {
        WeightConstraint.SHARP: SharpConstraint,
        WeightConstraint.SMOOTH: SmoothConstraint,
    }

    def __init__(self, layer_type: WeightTypes, constraint_type: WeightConstraint):
        assert isinstance(layer_type, WeightTypes) and isinstance(constraint_type, WeightConstraint)
        self.constrainer_class = self.CONSTRAINT_MAPPING[constraint_type]
        self.parametrization = self.constrainer_class(layer_type)

    @staticmethod
    def register_linear(linear: nn.Linear, parametrization: Constraint) -> nn.Module:
        """
        Should be called once upon construction of model, pytorch handles the subsequent constraints.
        :param linear: instance of nn.Linear which should have its 'weight' constrained.
        :param parametrization: nn.Module which takes weights and returns its constrained version.
        :return: instance of constrained nn.Linear.
        """
        assert isinstance(linear, nn.Linear)
        assert isinstance(parametrization, Constraint)
        return register_parametrization(linear, 'weight', parametrization)

    def register(self, module: nn.Module) -> nn.Module:
        """
        Register constraints for module.
        :param module: instance of nn.Module with multiple nn.Linear layers. For example: CustomRNNCell
        :return: instance of constrained nn.Module.
        """
        if hasattr(module, "weights_hh") and isinstance(module.weights_hh, nn.Linear):
            ConstraintRegistrar.register_linear(module.weights_hh, self.parametrization)
        if hasattr(module, "weights_ih_exc") and isinstance(module.weights_ih_exc, nn.Linear):
            ConstraintRegistrar.register_linear(module.weights_ih_exc, self.constrainer_class(WeightTypes.EXCITATORY))
        if hasattr(module, "weights_ih_inh") and isinstance(module.weights_ih_inh, nn.Linear):
            ConstraintRegistrar.register_linear(module.weights_ih_inh, self.constrainer_class(WeightTypes.EXCITATORY))
        return module
