"""
This script defines all classes used as weights constraints. These 
are typically used for determination of excitatory/inhibitory layer.
"""

import torch

from nn_model.type_variants import WeightTypes


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

    def hidden_constraint(self, module, kwargs):
        """
        Applies constraints on the `weight_hh` (hidden) parameters
        of the provided layer. Applying excitatory/inhibitory constraint.

        :param module: module (layer) to which we want to apply the constraint to.
        :param kwargs: kwargs of the `torch.clamp` function specifying the
        operation on the weights.
        """
        if hasattr(module, "W_hh"):
            module.W_hh.weight.data = torch.clamp(module.W_hh.weight.data, **kwargs)

    def input_constraint(self, module):
        """
        Applies constraint on the `weight_ih` (input) parameters of the provided
        layer. Applying excitatory/inhibitory constraint on the given
        part of the input of the layer.

        Applies the constrain in ascending order to the parts of the weights
        based on the properties of the input of the layer specified in the
        attribute `self.input_parameters`.

        :param module: module (layer) to which we want to apply the constraint to.
        """
        if hasattr(module, "W_ih"):
            end_index = 0
            for item in self.input_parameters:
                # Iterate each input layer part (input neuron layers).
                part_size = item["part_size"]
                part_kwargs = WeightConstraint.layer_kwargs[item["part_type"]]

                # Define the section where the constraint should be applied.
                start_index = end_index
                end_index += part_size

                # Apply constraint to the selected section of input weights.
                module.W_ih.weight.data[start_index:end_index] = torch.clamp(
                    module.W_ih.weight.data[start_index:end_index],
                    **part_kwargs,
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
            module, WeightConstraint.layer_kwargs[WeightTypes.EXCITATORY.value]
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
            module, WeightConstraint.layer_kwargs[WeightTypes.INHIBITORY.value]
        )
        self.input_constraint(module)
