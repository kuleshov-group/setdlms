
<h1 align="center">Planner Aware Path Learning in Diffusion Language Models Training</h1>

<p align="center">
  Fred Zhangzhi Peng, Zachary Bezemek, Jarrid Rector-Brooks, Shuibai Zhang,<br>
  Michael M. Bronstein, Anru Zhang, Alexander Tong, Joey Bose
</p>

<p align="center">
  <strong>ICLR 2026 (Oral)</strong>
</p>

PAPL is a simple planner-aligned modification of the standard masked discrete diffusion loss. The key idea is to reweight the per-token denoising loss using a **detached planner distribution** derived from the model’s own confidence, thereby reducing the mismatch between **uniform-random training paths** and **planner-based inference**.

This repo is intentionally minimal: it contains a single-file implementation of the PAPL loss for easy reading, reuse, and adaptation.

## Quick start


Then use the loss in your training code:

```python
import torch
from PAPL import papl_loss

B, L, V = 2, 8, 32

logits = torch.randn(B, L, V, requires_grad=True)
x0 = torch.randint(0, V, (B, L))
mask = torch.rand(B, L) < 0.5

loss, metrics = papl_loss(
    logits=logits,
    x0=x0,
    mask=mask,
    alpha=1.0,
    tau=1.0,
)

loss.backward()

print("loss:", loss.item())
print(metrics)
```

## Reference implementation

```python
def papl_loss(
    logits: torch.Tensor,
    x0: torch.Tensor,
    mask: torch.Tensor,
    alpha: float,
    tau: float,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the PAPL objective from Eq. (7)."""
    log_probs = F.log_softmax(logits, dim=-1)  # [B, L, V]

    # Log-probability of the correct clean token at each position.
    target_log_probs = log_probs.gather(dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # [B, L]
    target_nll = -target_log_probs

    # Only masked positions contribute to the loss.
    masked_nll = target_nll * mask

    # Detached planner: w_i ∝ exp(log p(correct token)/tau), normalized over masked positions.
    detached_scores = (target_log_probs.detach() / tau).masked_fill(~mask, float("-inf"))
    planner_weights = F.softmax(detached_scores, dim=-1)
    planner_weights = torch.where(mask, planner_weights, torch.zeros_like(planner_weights))

    # Per-example number of masked positions, matching 1 / (L-k) in the paper.
    n_masked = mask.sum(dim=-1).clamp_min(1).float()  # [B]
    base_weight = (1.0 / n_masked).unsqueeze(-1)      # [B, 1]

    weights = base_weight * (1.0 + alpha * planner_weights)
    loss_per_example = (weights * masked_nll).sum(dim=-1)
    loss = loss_per_example.mean()

    with torch.no_grad():
        metrics = {
            "loss": float(loss.item()),
            "avg_n_masked": float(n_masked.mean().item()),
            "avg_planner_entropy": float(
                (-(planner_weights.clamp_min(eps) * planner_weights.clamp_min(eps).log()).sum(dim=-1)).mean().item()
            ),
            "avg_correct_prob_on_masked": float(target_log_probs.exp()[mask].mean().item()) if mask.any() else 0.0,
        }

    return loss, metrics
```

## Notes

This implementation is designed to be:

* **minimal**, so the core idea is easy to inspect,
* **drop-in**, so it can replace a standard masked-token denoising loss with minimal changes,
* **faithful to the paper’s practical form**, where planner weights are detached.

A few practical details:

* `mask` should be boolean.
* Only masked positions contribute to the loss.
* The planner distribution is normalized only over masked positions.
* `tau < 1` makes the planner sharper; `tau > 1` makes it softer.
* Larger `alpha` places more emphasis on planner-aligned supervision.


It derives a planner-aware ELBO and then introduces PAPL as a simple and effective tractable objective.

## Code generation experiments

The **code generation experiments** from the paper are implemented in:

**github.com/pengzhangzhi/Open-dLLM**

If you are specifically interested in reproducing the paper’s coding results, please refer to that repository.

## Citation

If you use this code or the PAPL objective in your work, please cite:

```bibtex
@inproceedings{peng2026papl,
  title     = {Planner Aware Path Learning in Diffusion Language Models Training},
  author    = {Peng, Fred Zhangzhi and Bezemek, Zachary and Rector-Brooks, Jarrid and Zhang, Shuibai and Bronstein, Michael M. and Zhang, Anru and Bose, Joey and Tong, Alexander},
  booktitle = {International Conference on Learning Representations},
  year      = {2026}
}
```