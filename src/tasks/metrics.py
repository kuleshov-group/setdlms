import math
from collections.abc import Mapping

import torch
from torch import Tensor
from torchmetrics import Metric

LOG2 = math.log(2)


# noinspection LongLine
class Loss(Metric):
    # Adapted from https://docs.mosaicml.com/projects/composer/en/v0.14.1/_modules/composer/metrics/nlp.html#HFCrossEntropy  # noqa: E501

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.name = "loss"
        self.add_state("sum_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_batches", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, output: Mapping | Tensor, target: Tensor) -> None:
        loss = output["loss"]
        self.sum_loss += loss
        self.total_batches += 1

    def compute(self) -> Tensor:
        return self.sum_loss / self.total_batches


class NLL(Metric):
    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.name = "nll"
        self.add_state("mean_nll", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("weight", default=torch.tensor(0), dist_reduce_fx="sum")

    # noinspection LongLine
    def update(self, output: Mapping | Tensor, target: Tensor) -> None:
        value = output["nlls"]
        weight = output.get("tokens_mask", None)
        # TODO hack for logit shifting
        if weight is not None:
            weight = weight[:, -value.shape[1] :]

        # broadcast weight to value shape; copied from:
        # https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/aggregation.py#L501-L625  # noqa: E501
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        if weight is None:
            weight = torch.ones_like(value)
        elif not isinstance(weight, Tensor):
            weight = torch.as_tensor(weight, dtype=self.dtype, device=self.device)
        weight = torch.broadcast_to(weight, value.shape)

        if value.numel() == 0:
            return
        self.mean_nll += (value * weight).sum()
        self.weight += weight.sum()

    def compute(self) -> Tensor:
        return self.mean_nll / self.weight


class BPD(NLL):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.name = "bpd"

    def compute(self) -> Tensor:
        """Computes the bits per dimension.

        Returns:
          bpd
        """
        return self.mean_nll / self.weight / LOG2


class Perplexity(NLL):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.name = "ppl"

    def compute(self) -> Tensor:
        """Computes the Perplexity.

        Returns:
         Perplexity
        """
        return torch.exp(self.mean_nll / self.weight)
