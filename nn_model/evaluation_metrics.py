"""
This source code contains implementations of all evaluation metrics used
in our model.
"""

from typing import Tuple

import torch


class NormalizedCrossCorrelation:
    """
    Class for computation Normalized Cross Correlation defined in paper:
    https://www.biorxiv.org/content/10.1101/2023.03.21.533548v1.full.pdf

    The formula:
        CC_NORM = CC_ABS / CC_MAX

        CC_ABS = Cov(r^{dash}, y^{dash}) / sqrt(Var(r^{dash})*Var(y^{dash}))
        CC_MAX = sqrt((N*Var(r^{dash} - Var(y)^{dash}) / ((N-1)*Var(y^{dash})))

    Where `r` is in silico response (prediction), `y` is in vivo
    response (target), N is number of trials, and dashed values are means
    across the trials.
    """

    def __init__(self):
        """
        Initialized the shared variables needed for loss calculation.
        """
        self.batch_size: int = 0
        self.num_trials: int = 0
        self.time_duration: int = 0
        self.num_neurons: int = 0

    def _merge_neuron_time_dim(self, data_tensor: torch.Tensor) -> torch.Tensor:
        """
        Merges (reshapes) time and neuron dimension for metrics computation.

        :param data_tensor: tensor to be reshaped.
        :return: Returns reshaped tensor.
        """
        return data_tensor.view(
            self.batch_size, self.num_trials, self.time_duration * self.num_neurons
        )

    def _trials_avg(
        self, prediction: torch.Tensor, target: torch.Tensor, dim: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates prediction and target averages over the trial.

        :param prediction: prediction tensor.
        :param target: target tensor.
        :param dim: trials dimension (to compute average across).
        :return: Returns tuple of prediction and target tensor
        averaged over the trials.
        """
        avg_prediction = prediction.mean(dim=dim)
        avg_target = target.mean(dim=dim)

        return avg_prediction, avg_target

    def _neurons_std(
        self, avg_prediction: torch.Tensor, avg_target: torch.Tensor, dim: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates standard deviation over neurons for average
        predictions and targets over trials.

        :param avg_prediction: averaged prediction tensor across trials
        :param avg_target: averaged target tensor across trials.
        :param dim: dimension for which we want to compute
        the std (neurons dimension).
        :return: Returns tuple of tensors of std for predictions and targets.
        """
        std_pred = avg_prediction.std(dim=dim, unbiased=False)
        std_target = avg_target.std(dim=dim, unbiased=False)

        return std_pred, std_target

    def _covariance(
        self, avg_prediction: torch.Tensor, avg_target: torch.Tensor, dim: int = 1
    ) -> torch.Tensor:
        """
        Calculates the covariance between predictions and targets
        averaged across trials.

        :param avg_prediction: averaged prediction tensor across trials
        :param avg_target: averaged target tensor across trials.
        :param dim: dimension for which we want to compute
        the covariance (neurons dimension).
        :return: Returns covariance between predictions
        and targets averaged across trials.
        """
        # Mean across neurons.
        mean_pred = avg_prediction.mean(dim=dim, keepdim=True)
        mean_target = avg_target.mean(dim=dim, keepdim=True)

        # Calculate covariance between prediction and target
        return ((avg_prediction - mean_pred) * (avg_target - mean_target)).mean(dim=dim)

    def _cc_abs(
        self,
        avg_prediction: torch.Tensor,
        avg_target: torch.Tensor,
        std_pred: torch.Tensor,
        std_target: torch.Tensor,
        denom_offset: float = 1e-8,
    ) -> torch.Tensor:
        """
        Calculates CC_ABS from the paper. It represents Pearson's CC between
        predictions and targets averaged over trials.

        :param avg_prediction: averaged prediction tensor across trials
        :param avg_target: averaged target tensor across trials.
        :param std_pred: standard deviation of trial mean predictions across neurons.
        :param std_target: standard deviation of trial mean targets across neurons.
        :param denom_offset: offset for sanity of the division.
        :return: Returns the value of CC_ABS.
        """
        # Calculate covariance between predictions and targets.
        cov = self._covariance(avg_prediction, avg_target)

        # Calculate correlation coefficients:
        #   Cov(r^dash, y^dash) / sqrt(Var(r^dash) * Var(y^dash))
        denom = torch.clamp(
            std_pred * std_target, min=denom_offset
        )  # Assuring the denominator is reasonably large
        return cov / denom

    def _cc_max(
        self,
        target: torch.Tensor,
        var_target: torch.Tensor,
        num_trials: int,
        denom_offset: float = 1e-8,
    ) -> torch.Tensor:
        """
        Calculates CC_MAX value from the paper. It represents the upper bound
        of achievable performance given the the in vivo variability of the
        neuron and the number of trials.

        :param target: targets tensor.
        :param var_target: target variance over neurons.
        :param num_trials: number of trials.
        :param denom_offset: offset for sanity of the division.
        :return: returns the value of CC_MAX.
        """
        # Variance across trials.
        var_trials = target.var(dim=1, unbiased=False)

        # Mean variance across trials.
        mean_var_trials = var_trials.mean(dim=1)

        # N*Var(y^{dash}) - Var(y)^{dash}
        numerator = num_trials * var_target - mean_var_trials

        # (N-1)*Var(y^{dash})
        denominator = (num_trials - 1) * var_target

        cc_max = torch.sqrt(numerator / (denominator + denom_offset))

        return torch.clamp(cc_max, min=1e-6)  # Assuring the cc_max is reasonably large

    def _cc_norm(
        self,
        avg_prediction: torch.Tensor,
        avg_target: torch.Tensor,
        std_pred: torch.Tensor,
        std_target: torch.Tensor,
        target: torch.Tensor,
        denom_offset: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates CC_NORM from the paper.

        :param avg_prediction: averaged prediction tensor across trials
        :param avg_target: averaged target tensor across trials.
        :param std_pred: standard deviation of trial mean predictions across neurons.
        :param std_target: standard deviation of trial mean targets across neurons.
        :param target: target tensor.
        :param denom_offset: offset for sanity of the division.
        :return: Returns tuple of CC_NORM and CC_ABS (Pearson's CC).
        """
        cc_abs = self._cc_abs(avg_prediction, avg_target, std_pred, std_target)
        cc_max = self._cc_max(target, std_target * std_target, self.num_trials)

        cc_norm = cc_abs / (cc_max + denom_offset)

        # Return tuple of CC_NORM and CC_ABS (Pearson's CC)
        return torch.clamp(cc_norm, min=-1.0, max=1.0), torch.clamp(
            cc_abs, min=-1.0, max=1.0
        )

    def _batch_mean(self, cc_vector: torch.Tensor) -> float:
        """
        Calculates mean across batch for cross correlation values.

        :param cc_vector: values of CC for all batch examples.
        :return: Returns mean of given CC across batch.
        """
        return cc_vector.mean(dim=0).item()

    def calculate(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Calculates normalized cross correlation between predictions and targets.

        It expects 4D pytorch tensors as input of shape:
            `(batch_size, num_trials, time_duration, num_neurons)`

        Calculation is defined in paper:
        https://www.biorxiv.org/content/10.1101/2023.03.21.533548v1.full.pdf

        :param prediction: model predictions. In silico responses, `r` from the paper.
        :param target: target values. In vivo responses, `y` from the paper.
        :return: Returns tuple normalized and absolute (Pearson's CC) cross correlation value.
        """
        # Assuming prediction and target are PyTorch tensors.
        self.batch_size, self.num_trials, self.time_duration, self.num_neurons = (
            prediction.shape
        )

        # Reshape to have time and neurons in one dimension.
        prediction = self._merge_neuron_time_dim(prediction)
        target = self._merge_neuron_time_dim(target)

        # Calculate mean across the trials.
        avg_prediction, avg_target = self._trials_avg(prediction, target)

        # Calculate standard deviation across neurons.
        std_pred, std_target = self._neurons_std(avg_prediction, avg_target)

        # Calculate CC_NORM.
        cc_norm, cc_abs = self._cc_norm(
            avg_prediction,
            avg_target,
            std_pred,
            std_target,
            target,
        )

        return self._batch_mean(cc_norm), self._batch_mean(cc_abs)
