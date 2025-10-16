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

    def __init__(self, name="loss", update_key="loss", dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.name = name
        self.update_key = update_key
        self.add_state("sum_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_batches", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, output: Mapping | Tensor, target: Tensor) -> None:
        loss = output[self.update_key]
        self.sum_loss += loss
        self.total_batches += 1

    def compute(self) -> Tensor:
        return self.sum_loss / self.total_batches


class NLL(Metric):
    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(
        self,
        name="nll",
        update_key="nlls",
        weight_key="tokens_mask",
        dist_sync_on_step=False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.name = name
        self.update_key = update_key
        self.weight_key = weight_key
        self.add_state("mean_nll", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("weight", default=torch.tensor(0), dist_reduce_fx="sum")

    # noinspection LongLine
    def update(self, output: Mapping | Tensor, target: Tensor) -> None:
        value = output[self.update_key]
        weight = output.get(self.weight_key, None)
        if weight is not None:
            weight = weight[:, -value.shape[1] :]  # Logit shifting may create mismatch

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
        if self.weight.dtype != weight.dtype:
            self.weight = self.weight.to(weight.dtype)
        self.weight += weight.sum()

    def compute(self) -> Tensor:
        return self.mean_nll / self.weight


class BPD(NLL):
    def __init__(
        self,
        name="bpd",
        update_key="nlls",
        weight_key="tokens_mask",
        dist_sync_on_step=False,
    ):
        super().__init__(
            name=name,
            update_key=update_key,
            weight_key=weight_key,
            dist_sync_on_step=dist_sync_on_step,
        )

    def compute(self) -> Tensor:
        """Computes the bits per dimension.

        Returns:
          bpd
        """
        return self.mean_nll / self.weight / LOG2


class Perplexity(NLL):
    def __init__(
        self,
        name="ppl",
        update_key="nlls",
        weight_key="tokens_mask",
        dist_sync_on_step=False,
    ):
        super().__init__(
            name=name,
            update_key=update_key,
            weight_key=weight_key,
            dist_sync_on_step=dist_sync_on_step,
        )

    def compute(self) -> Tensor:
        """Computes the Perplexity.

        Returns:
         Perplexity
        """
        return torch.exp(self.mean_nll / self.weight)


class MaskedTokenFrequency(Metric):
    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(
        self,
        name="masked_token_frequency",
        update_key="masked_tokens",
        weight_key="tokens_mask",
        dist_sync_on_step=False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.name = name
        self.update_key = update_key
        self.weight_key = weight_key
        self.add_state("masked_tokens", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("weight", default=torch.tensor(0), dist_reduce_fx="sum")

    # noinspection LongLine
    def update(self, output: Mapping | Tensor, target: Tensor) -> None:
        value = output[self.update_key]
        weight = output.get(self.weight_key, None)

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
        self.masked_tokens += (value * weight).sum()
        self.weight += weight.sum()

    def compute(self) -> Tensor:
        return self.masked_tokens / self.weight
