"""Plain multiclass CE baseline model."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel


class MulticlassTopKPredictor(nn.Module):
    """Encode a question and produce one raw logit for each candidate k."""

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_labels: int = 15,
        dropout: float = 0.1,
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
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}")

        self.model_name = model_name
        self.num_labels = num_labels
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

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
        logits = self.classifier(self.dropout(question_representation))
        return {"logits": logits}
