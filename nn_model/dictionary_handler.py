"""
This script defines class that is used for manipulation with data stored in dictionaries.
Typically for the multiple layers results and variables.
"""

from typing import Tuple, Dict, List, Union, Any, Optional

import torch

from nn_model.models import PrimaryVisualCortexModel


class DictionaryHandler:
    """
    This class is used for manipulation with the dictionaries (especially with the model layers).
    """

    def __init__(self):
        pass

    @staticmethod
    def assign_args_kwargs(
        call_tuple: Tuple[Any, Optional[Tuple], Optional[Dict]]
    ) -> Tuple:
        """
        Assigns call args and kwargs in case they are not defined in the provided call tuple.

        :param call_tuple: Tuple of call object, its args and kwargs.
        :return: Tuple of called object, its args, and kwargs.
        """
        call_name = call_tuple[0]
        call_args = call_tuple[1] if len(call_tuple) > 1 else ()
        call_kwargs = call_tuple[2] if len(call_tuple) > 2 else {}

        return call_name, call_args, call_kwargs

    @staticmethod
    def apply_methods_calls(method_object, *methods_with_args):
        """
        Applies variable number of methods call on the given object.

        Example of usage of `methods_with_args`:

            Let's have list of tuples of function names and its parameters:
            ```python
            methods_to_call = [
                ('increment', (2,), {}),
                ('multiply', (3,), {}),
                ('get_value', (,), {print_value=True})
            ]
            ```

            Then:

            `value = DictionaryHandler.apply_methods_calls(value, *methods_to_call)`
            ```

            is alternative of calling:

            `value.increment(2).multiply(3).get_value(print_value=True)`

        :param method_object: Object on which we want to call the methods.
        :param methods_with_args: Ordered list of tuples where first value is method name and the
        rest are the arguments of the method. These methods should be called in this order on the
        provided object.

        NOTE: The methods needs to return the object values, otherwise it won't work.
        :return: Returns the input object after the application of the provided methods.
        """
        for method_with_args in methods_with_args:
            # method_name, method_args = method_with_args[0], methods_with_args[1:]
            method_name, method_args, method_kwargs = (
                DictionaryHandler.assign_args_kwargs(method_with_args)
            )
            method_object = getattr(method_object, method_name)(
                *method_args, **method_kwargs
            )

        return method_object

    @staticmethod
    def apply_function_calls(value, *functions_with_args):
        """
        Applies variable number of function calls on the provided object.

        Example of usage:
            Let's have list of tuples of function names, function args, and kwargs.
            ```python
            functions_to_call = [
                (increment, (2,), {}),
                (multiply, (3,), {}),
                (get_value, (,), {'print_results': True})
            ]
            ```

            Then:

            `result = DictionaryHandler.apply_function_calls(value, *functions_to_call)`

            is alternative of calling:

            `result = get_value(multiply((increment(value, 2)), 3) ,print_results=True)`

        :param value: Value on which we want to call the functions
        (its the argument of the first function).
        :param functions_with_args: Ordered list of tuples of the function objects as first
        value, then tuple of function args, and dictionary of function kwargs.
        These functions should be applied in the provided order.
        :return: Returns result of the application of the provided functions on the given value.
        """
        for function_with_args in functions_with_args:
            func, func_args, func_kwargs = DictionaryHandler.assign_args_kwargs(
                function_with_args
            )

            value = func(value, *func_args, **func_kwargs)

        return value

    @staticmethod
    def split_input_output_layers(
        layer_sizes: Dict[str, int],
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Splits layers sizes to input and output ones.

        :return: Returns tuple of dictionaries of input and output layer sizes.
        """
        input_layers = {
            key: layer_sizes[key] for key in PrimaryVisualCortexModel.input_layers
        }
        output_layers = {
            key: value for key, value in layer_sizes.items() if key not in input_layers
        }

        return input_layers, output_layers

    @staticmethod
    def slice_given_axes(
        data: torch.Tensor, slice_axes_dict: Dict[int, Union[int, slice]]
    ) -> torch.Tensor:
        """
        Slices given tensor in the provided axes on the provided positions.

        :param data: Tensor to be sliced.
        :param slice_axes_dict: Dictionary of axes as a key and slice positions as values.
        :return: Returns sliced tensor based on the provided arguments.
        """
        slices = []
        for i in range(data.dim()):
            if i in slice_axes_dict:
                slices.append(slice_axes_dict[i])
                continue
            slices.append(slice(None))

        return DictionaryHandler.slice_tensor(data, slices)

    @staticmethod
    def slice_tensor(
        data: torch.Tensor, slices: List[Union[int, slice]]
    ) -> torch.Tensor:
        """
        Slices the tensor based on the list of slices in each dimension.

        :param data: Tensor to be sliced.
        :param slices: List of slices in each dimension.
        :return: Returns sliced tensor based on the provided slices list.
        """
        return data[tuple(slices)]

    @staticmethod
    def apply_methods_on_dictionary(
        dictionary: Dict, methods_with_args: List[Tuple]
    ) -> Dict:
        """
        Applies provided methods on all dictionary values.

        :param dictionary: Dictionary on which we want to apply the methods.
        :param methods_with_args: List of methods with arguments
        (see `DictionaryHandler.apply_methods_calls` for more information).
        :return: Returns input dictionary after application of the provided
        methods on its values.
        """
        return {
            key: DictionaryHandler.apply_methods_calls(value, *methods_with_args)
            for key, value in dictionary.items()
        }

    @staticmethod
    def apply_functions_on_dictionary(
        dictionary: Dict, functions_with_args: List[Tuple]
    ) -> Dict:
        """
        Applies provided function calls on all dictionary values.

        :param dictionary: Dictionary on which we want to apply the methods.
        :param functions_with_args: List of functions with arguments
        (see `DictionaryHandler.apply_function_calls` for more information).
        :return: Returns input dictionary after application of the provided
        functions on its values.
        """
        return {
            key: DictionaryHandler.apply_function_calls(value, *functions_with_args)
            for key, value in dictionary.items()
        }
