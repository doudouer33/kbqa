"""Sequence-classification model metadata for stage 1."""

from typing import Dict


STAGE1_ID2LABEL: Dict[int, str] = {
    0: "topk_0",
    1: "topk_nonzero",
}
STAGE1_LABEL2ID: Dict[str, int] = {label: label_id for label_id, label in STAGE1_ID2LABEL.items()}


def load_stage1_model(model_name_or_path: str):
    """Load a HuggingFace sequence classifier with the stage-1 label space."""

    from transformers import AutoConfig, AutoModelForSequenceClassification

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=len(STAGE1_ID2LABEL),
        id2label=dict(STAGE1_ID2LABEL),
        label2id=dict(STAGE1_LABEL2ID),
    )
    config.num_labels = len(STAGE1_ID2LABEL)
    config.id2label = dict(STAGE1_ID2LABEL)
    config.label2id = dict(STAGE1_LABEL2ID)
    return AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)


def load_stage1_tokenizer(model_name_or_path: str):
    """Load the tokenizer paired with the stage-1 model."""

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

