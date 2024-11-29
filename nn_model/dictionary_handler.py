"""
This script defines class that is used for manipulation with data stored in dictionaries.
Typically for the multiple layers results and variables.
"""

from typing import Tuple, Dict, List

import torch

from nn_model.models import RNNCellModel


class DictionaryHandler:
    """
    This class is used for manipulation with the dictionaries (especially with the model layers).
    """

    def __init__(self):
        pass

    @staticmethod
    def split_input_output_layers(
        layer_sizes,
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Splits layers sizes to input and output ones.

        :return: Returns tuple of dictionaries of input and output layer sizes.
        """
        input_layers = {key: layer_sizes[key] for key in RNNCellModel.input_layers}
        output_layers = {
            key: value for key, value in layer_sizes.items() if key not in input_layers
        }

        return input_layers, output_layers

    @staticmethod
    def slice_given_axes(
        data: torch.Tensor, slice_axes_dict: Dict[int, int]
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
    def slice_tensor(data: torch.Tensor, slices: List) -> torch.Tensor:
        """
        Slices the tensor based on the list of slices in each dimension.

        :param data: Tensor to be sliced.
        :param slices: List of slices in each dimension.
        :return: Returns sliced tensor based on the provided slices list.
        """
        return data[tuple(slices)]


if __name__ == "__main__":
    data = torch.reshape(torch.arange(4 * 3 * 2 * 5), ((4, 3, 2, 5)))
    print(data)
    print(DictionaryHandler.slice_given_axes(data, {1: 1, 2: 0}))
    print(DictionaryHandler.slice_given_axes(data, {0: 1}))

    # print(DictionaryHandler.slice_given_axes(data, {1: 1, 2: 0}).shape)
