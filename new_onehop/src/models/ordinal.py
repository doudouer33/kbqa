"""Ordered-threshold ordinal model for dynamic Top-k prediction."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel


class OrdinalTopKPredictor(nn.Module):
    """Encode a question into one score and compare it with ordered thresholds."""

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_labels: int = 15,
        num_thresholds: int = 14,
        dropout: float = 0.1,
        threshold_init_min: float = -2.0,
        threshold_init_max: float = 2.0,
        threshold_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if not model_name:
            raise ValueError("model_name must be non-empty")
        if (
            isinstance(num_labels, bool)
            or not isinstance(num_labels, int)
            or num_labels <= 1
        ):
            raise ValueError(
                f"num_labels must be an integer greater than 1, got {num_labels!r}"
            )
        if num_thresholds != num_labels - 1:
            raise ValueError(
                "num_thresholds must equal num_labels - 1, "
                f"got {num_thresholds!r} and {num_labels!r}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}")
        for value, name in (
            (threshold_init_min, "threshold_init_min"),
            (threshold_init_max, "threshold_init_max"),
            (threshold_epsilon, "threshold_epsilon"),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric, got {value!r}")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if threshold_init_min >= threshold_init_max:
            raise ValueError("threshold_init_min must be less than threshold_init_max")
        if threshold_epsilon <= 0.0:
            raise ValueError("threshold_epsilon must be positive")

        self.model_name = model_name
        self.num_labels = num_labels
        self.num_thresholds = num_thresholds
        self.threshold_epsilon = float(threshold_epsilon)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.score_head = nn.Linear(hidden_size, 1)

        # Parameterize b_1 directly and every later gap as
        # softplus(raw_gap) + epsilon. This makes b_1 < ... < b_14 true by
        # construction throughout optimization. Internal gaps are initialized by
        # inverting softplus so the *actual* thresholds are evenly spaced over
        # the requested deterministic interval.
        initial_thresholds = torch.linspace(
            float(threshold_init_min),
            float(threshold_init_max),
            steps=num_thresholds,
            dtype=torch.float32,
        )
        desired_gaps = initial_thresholds[1:] - initial_thresholds[:-1]
        softplus_values = desired_gaps - self.threshold_epsilon
        if torch.any(softplus_values <= 0):
            raise ValueError(
                "Threshold initialization gaps must exceed threshold_epsilon"
            )
        raw_gaps = torch.log(torch.expm1(softplus_values))
        self.threshold_start = nn.Parameter(initial_thresholds[0].clone())
        self.raw_threshold_gaps = nn.Parameter(raw_gaps)

    def ordered_thresholds(self) -> torch.Tensor:
        """Return the strictly increasing actual threshold vector."""
        gaps = F.softplus(self.raw_threshold_gaps) + self.threshold_epsilon
        return torch.cat(
            (
                self.threshold_start.reshape(1),
                self.threshold_start + torch.cumsum(gaps, dim=0),
            )
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        question_representation = encoder_output.last_hidden_state[:, 0, :]
        score = self.score_head(self.dropout(question_representation)).squeeze(-1)
        thresholds = self.ordered_thresholds()
        ordinal_logits = score.unsqueeze(-1) - thresholds.unsqueeze(0)
        return {
            "ordinal_logits": ordinal_logits,
            "score": score,
            "thresholds": thresholds,
        }
