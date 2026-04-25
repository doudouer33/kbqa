"""BERT-based stage-2 bucket classifier for retrieval intensity."""

from typing import Dict, List, Sequence, Tuple, Union

import torch
from transformers import AutoConfig, AutoTokenizer, BertForSequenceClassification


STAGE2_BUCKET_DEFINITIONS: Dict[int, Dict[str, Union[str, List[int]]]] = {
    0: {
        "label_name": "topk_1_2",
        "topk_range": [1, 2],
        "description": "original top-k label in [1, 2]",
    },
    1: {
        "label_name": "topk_3_5",
        "topk_range": [3, 5],
        "description": "original top-k label in [3, 5]",
    },
    2: {
        "label_name": "topk_6_9",
        "topk_range": [6, 9],
        "description": "original top-k label in [6, 9]",
    },
    3: {
        "label_name": "topk_10_15",
        "topk_range": [10, 15],
        "description": "original top-k label in [10, 15]",
    },
}

STAGE2_ID2LABEL: Dict[int, str] = {
    bucket_id: bucket_info["label_name"] for bucket_id, bucket_info in STAGE2_BUCKET_DEFINITIONS.items()
}
STAGE2_LABEL2ID: Dict[str, int] = {label: idx for idx, label in STAGE2_ID2LABEL.items()}


def _build_stage2_config(model_name_or_path: str):
    config = AutoConfig.from_pretrained(model_name_or_path)
    if getattr(config, "model_type", None) != "bert":
        raise ValueError(
            f"Stage-2 classifier expects a BERT backbone, but got model_type={getattr(config, 'model_type', None)!r} "
            f"from {model_name_or_path!r}."
        )

    config.num_labels = 4
    config.id2label = dict(STAGE2_ID2LABEL)
    config.label2id = dict(STAGE2_LABEL2ID)
    return config


def build_stage2_model(model_name_or_path: str = "bert-base-uncased") -> BertForSequenceClassification:
    """
    Create a standard HuggingFace BERT sequence classifier for stage 2.

    BertForSequenceClassification already uses the usual dropout + linear head,
    so no extra custom head is required here.
    """

    config = _build_stage2_config(model_name_or_path)
    return BertForSequenceClassification.from_pretrained(model_name_or_path, config=config)


def load_stage2_tokenizer(model_name_or_path: str = "bert-base-uncased"):
    """Load the tokenizer paired with the BERT stage-2 classifier."""

    return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)


def load_stage2_model_and_tokenizer(
    model_name_or_path: str = "bert-base-uncased",
) -> Tuple[BertForSequenceClassification, object]:
    """Load both model and tokenizer from a pretrained name or saved checkpoint directory."""

    model = build_stage2_model(model_name_or_path=model_name_or_path)
    tokenizer = load_stage2_tokenizer(model_name_or_path=model_name_or_path)
    return model, tokenizer


@torch.inference_mode()
def predict_stage2_bucket(
    texts: Union[str, Sequence[str]],
    model: BertForSequenceClassification,
    tokenizer,
    max_length: int = 128,
    device: Union[str, torch.device, None] = None,
) -> List[Dict[str, object]]:
    """Small helper for stage-2 bucket inference on one or more questions."""

    if isinstance(texts, str):
        texts = [texts]

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
    model = model.to(device)
    model.eval()

    logits = model(**encoded).logits
    probabilities = torch.softmax(logits, dim=-1).cpu()
    predictions = probabilities.argmax(dim=-1).tolist()

    outputs: List[Dict[str, object]] = []
    for text, label_id, probability in zip(texts, predictions, probabilities):
        outputs.append(
            {
                "text": text,
                "bucket_id": int(label_id),
                "bucket_name": STAGE2_ID2LABEL[int(label_id)],
                "probabilities": probability.tolist(),
            }
        )
    return outputs

