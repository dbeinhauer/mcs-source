from typing import Dict, List, Tuple
from pathlib import Path

import pickle
import torch

import nn_model.globals


class VisibleNeuronsHandler:
    """
    Class handling the differentiation between visible and invisible neurons in the model.
    """

    def __init__(self, arguments):
        self.visible_neurons_ratio = arguments.visible_neurons_ratio
        self.visible_neurons_mask_1d = None
        if self.visible_neurons_ratio < 1.0:
            # Load indices only if ratio is subset of the full model.
            self.visible_neurons_mask_1d = self._load_visible_neuron_indices()

    @staticmethod
    def get_visible_indices_path(
        visible_neurons_ratio,
        size_multiplier=nn_model.globals.SIZE_MULTIPLIER * 100,
        directory_path: str = "",
    ) -> Tuple[str, str]:
        """
        Returns the path to the visible neurons indices file based on the ratio and size multiplier.

        :param visible_neurons_ratio: Ratio of visible neurons.
        :param size_multiplier: Size multiplier for the model.
        :return: Path to the visible neurons indices file.
        """
        directory_path = (
            nn_model.globals.DEFAULT_PATHS[
                nn_model.globals.PathDefaultFields.VISIBLE_NEURONS_DIR.value
            ]
            if directory_path == ""
            else directory_path
        )
        filename = (
            f"size_{int(size_multiplier)}_ratio_{int(visible_neurons_ratio*100)}.pkl"
        )
        return directory_path, filename

    @staticmethod
    def _create_visible_neurons_mask(
        indices: Dict[str, List[int]],
    ) -> Dict[str, torch.Tensor]:
        """
        Creates a mask for visible neurons based on the provided indices.

        :param indices: Indices of the visible neurons for each layer.
        :return: Returns a 1D mask for each layer where visible neurons are set to True.
        """
        mask_layers = {}
        for layer, data in indices.items():
            mask_1d = torch.zeros(nn_model.globals.MODEL_SIZES[layer], dtype=torch.bool)
            mask_1d[data] = True
            mask_layers[layer] = mask_1d

        return mask_layers

    def _load_visible_neuron_indices(self) -> Dict[str, torch.Tensor]:
        """
        Loads indices of the visible neurons for each layer.

        :return:
        """
        directory, filename = self.get_visible_indices_path(
            self.visible_neurons_ratio,
        )
        with open(Path(directory) / filename, "rb") as pickle_file:
            return VisibleNeuronsHandler._create_visible_neurons_mask(
                pickle.load(pickle_file)
            )

    @staticmethod
    def _expand_mask_to_match_last_dim(
        mask_1d: torch.Tensor, reference_tensor: torch.Tensor
    ):
        """
        Expands a 1D mask to match the shape of a reference tensor,
        assuming the mask applies to the last dimension.

        :param: mask_1d: shape [D]
        :param: reference_tensor: any shape [..., D]
        :return: expanded_mask: shape [..., D]
        """
        ref_shape = reference_tensor.shape
        expand_shape = [1] * (len(ref_shape) - 1) + [mask_1d.size(0)]
        expanded_mask = mask_1d.view(*expand_shape).expand(*ref_shape)
        return expanded_mask

    def _expand_masks_to_responses(
        self, responses_tensor: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Expands the 1D mask to match the shape of the responses tensor.

        :param responses_tensor: Tensor to match the map shape with.
        :return: Returns the expanded mask that matches the shape of the responses tensor.
        """
        return {
            layer: VisibleNeuronsHandler._expand_mask_to_match_last_dim(
                self.visible_neurons_mask_1d[layer], responses_tensor[layer]
            )
            for layer in responses_tensor
        }

    def _select_on_mask(
        self, data_dict: Dict[str, torch.Tensor], mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Selects data based on the provided mask.

        :param data: Data tensor to select from for each layer.
        :param mask: Mask tensor to apply.
        :return: Returns the selected data based on the mask for each layer.
        The not selected data are kept as zeros (the original shape is preserved).
        """
        return {
            layer: data_dict[layer][..., mask[layer]]  # assuming same shape logic
            for layer in data_dict
        }

    def assign_teacher_forced_responses(
        self,
        target_responses: Dict[str, torch.Tensor],
        prediction_responses: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Takes all target responses and assigns them to the visible neurons.
        The rest of the neurons are "invisible" and this the previous prediction
        responses are returned.

        :param teacher_forced_neurons: Target responses of the neurons (to be used as
        teacher forcing neurons).
        :param invisible_neurons: Prediction responses of the neurons (to be used for the
        invisible neurons).
        :return: Returns neuron responses with the visible and invisible neurons properly assigned.
        """
        if not self.visible_neurons_mask_1d:
            # We have all neurons visible, return the original visible neurons.
            return target_responses

        expanded_mask = self._expand_masks_to_responses(target_responses)
        expanded_mask = {
            layer: expanded_mask[layer].to(nn_model.globals.DEVICE)
            for layer in expanded_mask
        }

        # Assign target responses to visible neurons. Function `torch.where` will
        # ensure the correct gradient flow for `prediction_responses`.
        return {
            layer: torch.where(
                expanded_mask[layer],
                target_responses[layer],
                prediction_responses[layer],  # .detach(),
            )
            for layer in target_responses
        }

    def split_visible_invisible_neurons(
        self, neuron_responses: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Selects only the visible neurons from the neuron responses.

        :param neuron_responses: Responses of the neurons.
        :return: Returns tuple of the visible neurons and invisible neurons from the neuron responses.
        """
        if not self.visible_neurons_mask_1d:
            return neuron_responses, {}

        visible_neurons = self._select_on_mask(
            neuron_responses, self.visible_neurons_mask_1d
        )
        invisible_neurons = self._select_on_mask(
            neuron_responses,
            {layer: ~mask for layer, mask in self.visible_neurons_mask_1d.items()},
        )

        return visible_neurons, invisible_neurons
