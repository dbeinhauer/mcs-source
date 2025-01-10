"""
This script defines all classes used as weights constraints. These 
are typically used for determination of excitatory/inhibitory layer.
"""

import torch

from nn_model.type_variants import WeightTypes, LayerConstraintFields


class WeightConstraint:
    """
    Parent class used for applying weight constraints for
    excitatory/inhibitory layers.
    """

    # kwargs of the excitatory/inhibitory layers.
    layer_kwargs = {
        WeightTypes.EXCITATORY.value: {"min": 0},
        WeightTypes.INHIBITORY.value: {"max": 0},
    }

    def __init__(self, input_parameters):
        """
        Assigns information used for weight application.
        :param input_parameters: list of dictionaries that describes the
        input constitution of the layer based on which we can select correct
        constraints.

        The list is sorted in ascending order. It means that
        the first item of the list defines first part of the layer
        (the lowest indices of the layer). Then follows the next neurons in the
        layer till it reaches the end of the input.

        In the directory there should be keys `part_type` specifying which
        type of neurons belongs to corresponding partition (excitatory/inhibitory),
        possible values are keys from `layer_kwargs` (it means `['inh', 'exc']`),
        and `part_size` defining number of neurons that belongs to such part.
        The sum of `part_size` values should be equal to input size of this layer.
        """
        # List of dictionaries of each input layer of this layer information.
        self.input_parameters = input_parameters

    def hidden_constraint(self, module, layer_type, kwargs):
        """
        Applies constraints on the `weight_hh` (self-recurrent) parameters
        of the provided layer. Applying excitatory/inhibitory constraint.

        :param module: module (layer) to which we want to apply the constraint to.
        :param layer_type: Identifier whether the layer is inhibitory or excitatory.
        :param kwargs: kwargs of the `torch.clamp` function specifying the
        operation on the weights.
        """
        assert layer_type in [
            WeightTypes.EXCITATORY.value,
            WeightTypes.INHIBITORY.value,
        ]
        if hasattr(module, "weights_hh"):
            module.weights_hh.weight.data = torch.clamp(
                module.weights_hh.weight.data, **kwargs
            )

    def input_constraint(self, module):
        """
        Applies constraints on the input weights of the provided layer.
        Differentiates between excitatory/inhibitory layers.

        :param module: Module to apply the weight on.
        """
        if hasattr(module, "weights_ih_exc"):
            module.weights_ih_exc.weight.data = torch.clamp(
                module.weights_ih_exc.weight.data,
                **WeightConstraint.layer_kwargs[WeightTypes.EXCITATORY.value],
            )
        if hasattr(module, "weights_ih_inh"):
            module.weights_ih_inh.weight.data = torch.clamp(
                module.weights_ih_inh.weight.data,
                **WeightConstraint.layer_kwargs[WeightTypes.INHIBITORY.value],
            )


class ExcitatoryWeightConstraint(WeightConstraint):
    """
    Class used for applying weight constraints for excitatory layers.
    """

    def __init__(self, input_parameters):
        """
        :param input_parameters: input parameters for the parent
        `WeightConstraint` class.
        """
        super().__init__(input_parameters)

    def apply(self, module):
        """
        Applies the constraints on the given module.

        :param module: module (layer) to which we want to apply the constraint to.
        """
        # Apply excitatory condition to all hidden neurons.
        self.hidden_constraint(
            module,
            WeightTypes.EXCITATORY.value,
            WeightConstraint.layer_kwargs[WeightTypes.EXCITATORY.value],
        )
        self.input_constraint(module)


class InhibitoryWeightConstraint(WeightConstraint):
    """
    Class used for applying weight constraints for excitatory layers.
    """

    def __init__(self, input_parameters):
        """
        :param input_parameters: input parameters for the parent
        `WeightConstraint` class.
        """
        super().__init__(input_parameters)

    def apply(self, module):
        """
        Applies the constraints on the given module.

        :param module: module (layer) to which we want to apply the constraint to.
        """
        # Apply inhibitory condition to all hidden neurons.
        self.hidden_constraint(
            module,
            WeightTypes.INHIBITORY.value,
            WeightConstraint.layer_kwargs[WeightTypes.INHIBITORY.value],
        )
        self.input_constraint(module)
