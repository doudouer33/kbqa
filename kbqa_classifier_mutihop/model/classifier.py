"""Sequence-classification model metadata for multihop top-k labels 1-8."""

from typing import Dict


LABEL_VALUES = tuple(range(1, 9))
CLASS_ID2LABEL: Dict[int, str] = {label_id - 1: f"label_{label_id}" for label_id in LABEL_VALUES}
CLASS_LABEL2ID: Dict[str, int] = {label: label_id for label_id, label in CLASS_ID2LABEL.items()}
CLASS_ID_TO_VALUE: Dict[int, int] = {label_id - 1: label_id for label_id in LABEL_VALUES}
CLASS_VALUE_TO_ID: Dict[int, int] = {label_id: label_id - 1 for label_id in LABEL_VALUES}


def load_classifier_model(model_name_or_path: str):
    """Load a HuggingFace sequence classifier with the multihop label space."""

    from transformers import AutoConfig, AutoModelForSequenceClassification

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=len(CLASS_ID2LABEL),
        id2label=dict(CLASS_ID2LABEL),
        label2id=dict(CLASS_LABEL2ID),
    )
    config.num_labels = len(CLASS_ID2LABEL)
    config.id2label = dict(CLASS_ID2LABEL)
    config.label2id = dict(CLASS_LABEL2ID)
    return AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)


def load_classifier_tokenizer(model_name_or_path: str):
    """Load the tokenizer paired with the classifier model."""

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
