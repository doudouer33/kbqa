"""BERT-based stage-1 binary classifier for retrieval necessity."""

from typing import Dict, List, Sequence, Tuple, Union

import torch
from transformers import AutoConfig, AutoTokenizer, BertForSequenceClassification


STAGE1_ID2LABEL: Dict[int, str] = {
    0: "no_retrieval",
    1: "needs_retrieval",
}

STAGE1_LABEL2ID: Dict[str, int] = {label: idx for idx, label in STAGE1_ID2LABEL.items()}


def _build_stage1_config(model_name_or_path: str):
    config = AutoConfig.from_pretrained(model_name_or_path)
    if getattr(config, "model_type", None) != "bert":
        raise ValueError(
            f"Stage-1 classifier expects a BERT backbone, but got model_type={getattr(config, 'model_type', None)!r} "
            f"from {model_name_or_path!r}."
        )

    config.num_labels = 2
    config.id2label = dict(STAGE1_ID2LABEL)
    config.label2id = dict(STAGE1_LABEL2ID)
    return config


def build_stage1_model(model_name_or_path: str = "bert-base-uncased") -> BertForSequenceClassification:
    """Create a standard HuggingFace sequence-classification model for stage 1."""

    config = _build_stage1_config(model_name_or_path)
    return BertForSequenceClassification.from_pretrained(model_name_or_path, config=config)


def load_stage1_tokenizer(model_name_or_path: str = "bert-base-uncased"):
    """Load the tokenizer paired with the BERT stage-1 classifier."""

    return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)


def load_stage1_model_and_tokenizer(
    model_name_or_path: str = "bert-base-uncased",
) -> Tuple[BertForSequenceClassification, object]:
    """Load both model and tokenizer from a pretrained name or saved checkpoint directory."""

    model = build_stage1_model(model_name_or_path=model_name_or_path)
    tokenizer = load_stage1_tokenizer(model_name_or_path=model_name_or_path)
    return model, tokenizer


@torch.inference_mode()
def predict_stage1(
    texts: Union[str, Sequence[str]],
    model: BertForSequenceClassification,
    tokenizer,
    max_length: int = 128,
    device: Union[str, torch.device, None] = None,
) -> List[Dict[str, object]]:
    """Small helper for stage-1 inference on one or more questions."""

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
                "label_id": int(label_id),
                "label_name": STAGE1_ID2LABEL[int(label_id)],
                "probabilities": probability.tolist(),
            }
        )
    return outputs

