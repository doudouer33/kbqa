"""Focused Phase 2 unit and numerical sanity checks."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from new_onehop.src.losses import (
    build_ordinal_targets,
    ordinal_bce_loss,
    ordinal_cumulative_to_class_probs,
)
from new_onehop.src.models.ordinal import OrdinalTopKPredictor


class _DummyEncoder(nn.Module):
    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(32, hidden_size)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> SimpleNamespace:
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class OrdinalTest(unittest.TestCase):
    def test_ordinal_targets_examples(self) -> None:
        targets = build_ordinal_targets(torch.tensor([0, 4, 14], dtype=torch.long))
        self.assertEqual(targets.shape, (3, 14))
        self.assertEqual(targets.dtype, torch.float32)
        self.assertEqual(targets[0].tolist(), [0.0] * 14)
        self.assertEqual(targets[1].tolist(), [1.0] * 4 + [0.0] * 10)
        self.assertEqual(targets[2].tolist(), [1.0] * 14)

    def test_thresholds_probabilities_and_optimizer_step(self) -> None:
        with patch(
            "new_onehop.src.models.ordinal.AutoModel.from_pretrained",
            return_value=_DummyEncoder(),
        ):
            model = OrdinalTopKPredictor(model_name="dummy")
        thresholds = model.ordered_thresholds().detach()
        self.assertEqual(thresholds.shape, (14,))
        self.assertTrue(torch.all(thresholds[1:] > thresholds[:-1]))
        self.assertTrue(torch.allclose(thresholds[[0, -1]], torch.tensor([-2.0, 2.0])))

        input_ids = torch.randint(0, 32, (4, 6))
        attention_mask = torch.ones_like(input_ids)
        labels = torch.tensor([0, 4, 8, 14], dtype=torch.long)
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        self.assertEqual(output["ordinal_logits"].shape, (4, 14))
        targets = build_ordinal_targets(labels)
        loss = ordinal_bce_loss(output["ordinal_logits"], targets)
        self.assertTrue(torch.isfinite(loss))

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss.backward()
        optimizer.step()
        updated_thresholds = model.ordered_thresholds().detach()
        self.assertTrue(torch.all(updated_thresholds[1:] > updated_thresholds[:-1]))

        updated_output = model(input_ids=input_ids, attention_mask=attention_mask)
        cumulative = torch.sigmoid(updated_output["ordinal_logits"])
        self.assertTrue(torch.all(cumulative[:, :-1] >= cumulative[:, 1:]))
        class_probabilities = ordinal_cumulative_to_class_probs(cumulative)
        self.assertEqual(class_probabilities.shape, (4, 15))
        self.assertTrue(torch.isfinite(class_probabilities).all())
        self.assertTrue(torch.all(class_probabilities >= 0.0))
        self.assertTrue(
            torch.allclose(
                class_probabilities.sum(dim=1),
                torch.ones(4),
                atol=1e-6,
                rtol=0.0,
            )
        )
        predicted_k = torch.argmax(class_probabilities, dim=1) + 1
        self.assertTrue(torch.all((predicted_k >= 1) & (predicted_k <= 15)))

    def test_non_monotonic_cumulative_probabilities_fail(self) -> None:
        cumulative = torch.linspace(0.9, 0.1, 14).unsqueeze(0)
        cumulative[0, 7] = cumulative[0, 6] + 0.1
        with self.assertRaisesRegex(RuntimeError, "monotonically"):
            ordinal_cumulative_to_class_probs(cumulative)


if __name__ == "__main__":
    unittest.main()
