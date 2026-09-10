"""Plain ordinal BCE targets, loss, and cumulative probability conversion."""

from __future__ import annotations

import torch
from torch import nn


DEFAULT_NUM_THRESHOLDS = 14
PROBABILITY_TOLERANCE = 1e-6
PROBABILITY_SUM_TOLERANCE = 1e-5


def build_ordinal_targets(
    labels: torch.Tensor,
    num_thresholds: int = DEFAULT_NUM_THRESHOLDS,
) -> torch.Tensor:
    """Build y_j = 1[k*>j] from zero-based labels (hard_k = label + 1)."""
    if not isinstance(labels, torch.Tensor):
        raise TypeError("labels must be a torch.Tensor")
    if labels.ndim != 1:
        raise ValueError(f"labels must have shape [B], got {tuple(labels.shape)}")
    if labels.dtype == torch.bool or labels.is_floating_point():
        raise ValueError(f"labels must have an integer dtype, got {labels.dtype}")
    if (
        isinstance(num_thresholds, bool)
        or not isinstance(num_thresholds, int)
        or num_thresholds <= 0
    ):
        raise ValueError("num_thresholds must be a positive integer")
    if labels.numel() == 0:
        raise ValueError("labels must be non-empty")
    invalid = (labels < 0) | (labels > num_thresholds)
    if torch.any(invalid):
        invalid_values = labels[invalid].detach().cpu().tolist()
        raise ValueError(
            f"zero-based labels must be in [0, {num_thresholds}], got {invalid_values[:10]}"
        )
    threshold_indices = torch.arange(
        1, num_thresholds + 1, device=labels.device, dtype=labels.dtype
    )
    return (labels.unsqueeze(1) >= threshold_indices.unsqueeze(0)).to(torch.float32)


def ordinal_bce_loss(
    ordinal_logits: torch.Tensor,
    ordinal_targets: torch.Tensor,
) -> torch.Tensor:
    """Unweighted BCEWithLogitsLoss with mean reduction over B x thresholds."""
    if ordinal_logits.ndim != 2:
        raise ValueError(
            "ordinal_logits must have shape [B, num_thresholds], "
            f"got {tuple(ordinal_logits.shape)}"
        )
    if ordinal_targets.ndim != 2:
        raise ValueError(
            "ordinal_targets must have shape [B, num_thresholds], "
            f"got {tuple(ordinal_targets.shape)}"
        )
    if ordinal_logits.shape != ordinal_targets.shape:
        raise ValueError(
            f"ordinal logits/target shape mismatch: {tuple(ordinal_logits.shape)} "
            f"vs {tuple(ordinal_targets.shape)}"
        )
    if ordinal_logits.numel() == 0:
        raise ValueError("ordinal logits and targets must be non-empty")
    if (
        not ordinal_logits.is_floating_point()
        or not ordinal_targets.is_floating_point()
    ):
        raise ValueError("ordinal logits and targets must have floating dtypes")
    if not torch.isfinite(ordinal_logits).all():
        raise RuntimeError("ordinal_logits contains NaN or Inf")
    if not torch.isfinite(ordinal_targets).all():
        raise RuntimeError("ordinal_targets contains NaN or Inf")
    if torch.any((ordinal_targets < 0.0) | (ordinal_targets > 1.0)):
        raise ValueError("ordinal_targets must be in [0, 1]")
    # Intentionally plain/unweighted: Phase 2 controls variables by using no
    # weight, pos_weight, focal term, or label smoothing.
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    loss = criterion(ordinal_logits, ordinal_targets.to(dtype=ordinal_logits.dtype))
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite ordinal BCE loss: {loss.item()}")
    return loss


def ordinal_cumulative_to_class_probs(
    cumulative_probabilities: torch.Tensor,
    *,
    negative_tolerance: float = PROBABILITY_TOLERANCE,
    sum_tolerance: float = PROBABILITY_SUM_TOLERANCE,
) -> torch.Tensor:
    """Convert P(k>j), j=1..14, into the categorical P(k), k=1..15."""
    if cumulative_probabilities.ndim != 2:
        raise ValueError(
            "cumulative_probabilities must have shape [B, num_thresholds], "
            f"got {tuple(cumulative_probabilities.shape)}"
        )
    if cumulative_probabilities.shape[1] <= 0:
        raise ValueError("cumulative_probabilities must contain thresholds")
    if not cumulative_probabilities.is_floating_point():
        raise ValueError("cumulative_probabilities must have a floating dtype")
    if not torch.isfinite(cumulative_probabilities).all():
        raise RuntimeError("cumulative_probabilities contains NaN or Inf")
    if torch.any(cumulative_probabilities < -negative_tolerance) or torch.any(
        cumulative_probabilities > 1.0 + negative_tolerance
    ):
        raise RuntimeError("cumulative probabilities fall significantly outside [0, 1]")
    monotonic_differences = (
        cumulative_probabilities[:, :-1] - cumulative_probabilities[:, 1:]
    )
    if torch.any(monotonic_differences < -negative_tolerance):
        raise RuntimeError(
            "Cumulative probabilities are not monotonically non-increasing"
        )

    raw_class_probabilities = torch.cat(
        (
            1.0 - cumulative_probabilities[:, :1],
            monotonic_differences,
            cumulative_probabilities[:, -1:],
        ),
        dim=1,
    )
    if not torch.isfinite(raw_class_probabilities).all():
        raise RuntimeError("Converted class probabilities contain NaN or Inf")
    if torch.any(raw_class_probabilities < -negative_tolerance):
        raise RuntimeError(
            "Converted class probabilities are significantly negative; "
            "this indicates a cumulative-monotonicity bug"
        )
    probability_sums = raw_class_probabilities.sum(dim=1)
    max_sum_error = torch.max(torch.abs(probability_sums - 1.0)).item()
    if max_sum_error > sum_tolerance:
        raise RuntimeError(
            f"Converted class probabilities do not sum to 1; max error={max_sum_error:.8g}"
        )

    # Only remove floating-point dust, then apply a correspondingly tiny
    # normalization. Significant negatives and sum errors already failed above.
    class_probabilities = raw_class_probabilities.clamp(min=0.0)
    class_probabilities = class_probabilities / class_probabilities.sum(
        dim=1, keepdim=True
    )
    if not torch.isfinite(class_probabilities).all():
        raise RuntimeError("Normalized class probabilities contain NaN or Inf")
    return class_probabilities
