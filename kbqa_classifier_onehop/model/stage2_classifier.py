"""Sequence-classification model metadata for stage 2."""

from typing import Dict, List, Union


STAGE2_BUCKET_DEFINITIONS: Dict[int, Dict[str, Union[str, List[int]]]] = {
    0: {
        "label_name": "topk_1_2",
        "topk_range": [1, 2],
    },
    1: {
        "label_name": "topk_3_5",
        "topk_range": [3, 5],
    },
    2: {
        "label_name": "topk_6_9",
        "topk_range": [6, 9],
    },
    3: {
        "label_name": "topk_10_15",
        "topk_range": [10, 15],
    },
}
STAGE2_ID2LABEL: Dict[int, str] = {
    bucket_id: str(bucket_info["label_name"])
    for bucket_id, bucket_info in STAGE2_BUCKET_DEFINITIONS.items()
}
STAGE2_LABEL2ID: Dict[str, int] = {label: label_id for label_id, label in STAGE2_ID2LABEL.items()}


def load_stage2_model(model_name_or_path: str):
    """Load a HuggingFace sequence classifier with the stage-2 bucket space."""

    from transformers import AutoConfig, AutoModelForSequenceClassification

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=len(STAGE2_ID2LABEL),
        id2label=dict(STAGE2_ID2LABEL),
        label2id=dict(STAGE2_LABEL2ID),
    )
    config.num_labels = len(STAGE2_ID2LABEL)
    config.id2label = dict(STAGE2_ID2LABEL)
    config.label2id = dict(STAGE2_LABEL2ID)
    return AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)


def load_stage2_tokenizer(model_name_or_path: str):
    """Load the tokenizer paired with the stage-2 model."""

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

