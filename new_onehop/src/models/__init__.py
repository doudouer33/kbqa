"""Model heads for dynamic Top-k prediction."""

from .multiclass import MulticlassTopKPredictor
from .ordinal import OrdinalTopKPredictor

__all__ = ["MulticlassTopKPredictor", "OrdinalTopKPredictor"]
