"""
This source code contains implementations of all evaluation metrics used
in our model.
"""

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
        self.batch_size = 0
        self.num_trials = 0
        self.time_duration = 0
        self.num_neurons = 0

    def _convert_prediction_to_spikes(self, prediction):
        """
        Rounds the prediction to the closes integer (should represent spikes).

        :param prediction: torch tensor of float prediction.
        :return: Returns predictions rounded to integer.
        """
        return prediction.round()

    def _merge_neuron_time_dim(self, data_tensor):
        """
        Merges (reshapes) time and neuron dimension for metrics computation.

        :param data_tensor: tensor to be reshaped.
        :return: Returns reshaped tensor.
        """
        return data_tensor.view(
            self.batch_size, self.num_trials, self.time_duration * self.num_neurons
        )

    def _trials_avg(self, prediction, target, dim: int = 1):
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

    def _neurons_std(self, avg_prediction, avg_target, dim: int = 1):
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

    def _covariance(self, avg_prediction, avg_target, dim: int = 1) -> float:
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
        avg_prediction,
        avg_target,
        std_pred,
        std_target,
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

        # Calculate correlation coefficients:
        #   Cov(r^dash, y^dash) / sqrt(Var(r^dash) * Var(y^dash))
        # return cov / ((std_pred * std_target) + denom_offset)

    def _cc_max(
        self, target, var_target, num_trials: int, denom_offset: float = 1e-8
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
        avg_prediction,
        avg_target,
        std_pred,
        std_target,
        target,
        denom_offset: float = 1e-8,
    ) -> torch.Tensor:
        """
        Calculates CC_NORM from the paper.

        :param avg_prediction: averaged prediction tensor across trials
        :param avg_target: averaged target tensor across trials.
        :param std_pred: standard deviation of trial mean predictions across neurons.
        :param std_target: standard deviation of trial mean targets across neurons.
        :param target: target tensor.
        :param denom_offset: offset for sanity of the division.
        :return: Returns value of CC_NORM.
        """
        cc_abs = self._cc_abs(avg_prediction, avg_target, std_pred, std_target)
        cc_max = self._cc_max(target, std_target * std_target, self.num_trials)

        cc_norm = cc_abs / (cc_max + denom_offset)

        # return cc_abs / (cc_max + denom_offset)
        return torch.clamp(cc_norm, min=-1.0, max=1.0)

    def _batch_mean(self, cc_norm):
        """
        Calculates mean across batch for CC_NORM values.

        :param cc_norm: values of CC_NORM for all batch examples.
        :return: Returns mean of CC_NORM across batch.
        """
        return cc_norm.mean(dim=0).item()

    def calculate(self, prediction, target) -> float:
        """
        Calculates normalized cross correlation between predictions and targets.

        It expects 4D pytorch tensors as input of shape:
            `(batch_size, num_trials, time_duration, num_neurons)`

        Calculation is defined in paper:
        https://www.biorxiv.org/content/10.1101/2023.03.21.533548v1.full.pdf

        :param prediction: model predictions. In silico responses, `r` from the paper.
        :param target: target values. In vivo responses, `y` from the paper.
        :return: Returns normalized cross correlation value.
        """
        # Assuming prediction and target are PyTorch tensors.
        self.batch_size, self.num_trials, self.time_duration, self.num_neurons = (
            prediction.shape
        )

        # Round the prediction to closest integer.
        # prediction = self._convert_prediction_to_spikes(prediction)

        # Reshape to have time and neurons in one dimension.
        prediction = self._merge_neuron_time_dim(prediction)
        target = self._merge_neuron_time_dim(target)

        # Calculate mean across the trials.
        avg_prediction, avg_target = self._trials_avg(prediction, target)

        # Calculate standard deviation across neurons.
        std_pred, std_target = self._neurons_std(avg_prediction, avg_target)

        # Calculate CC_NORM.
        cc_norm = self._cc_norm(
            avg_prediction,
            avg_target,
            std_pred,
            std_target,
            target,
        )

        return self._batch_mean(cc_norm)
