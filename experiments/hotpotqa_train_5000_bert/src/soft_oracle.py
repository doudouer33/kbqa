"""Original Soft Oracle pure core, copied from build_soft_oracle.py (not imported).
Scheme D: F1 >= max_all_k(F1)-epsilon AND sentence_recall_lower >= tau.
Cost is divided by the question's all-k maximum; utility=F1-lambda*cost.
Masked softmax(utility/T); all-zero and empty-candidate labels remain null.
Source hashes in ../reports/algorithm_sources.json.
"""
import math
from dataclasses import dataclass, asdict
BOUNDARY_ATOL = 1e-12
SCHEMES = {'A':(False,False),'B':(True,False),'C':(False,True),'D':(True,True)}
COST_METRICS = ('context_tokens','prompt_tokens','completion_tokens','total_tokens')
EVIDENCE_METRICS = tuple(u+'_recall_'+b for u in ('sentence','paragraph') for b in ('lower','upper'))

def require(condition, message):
    if not condition:
        raise ValueError(message)

def number(value, name, low=None, high=None, integer=False):
    require(not isinstance(value, bool), "Boolean instead of number: " + name)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Invalid number: " + name)
    require(math.isfinite(result), "Non-finite number: " + name)
    require(low is None or result >= low, "Below range: " + name)
    require(high is None or result <= high, "Above range: " + name)
    if integer:
        require(result.is_integer(), "Non-integer: " + name)
        return int(result)
    return result

@dataclass(frozen=True)
class Parameters:
    epsilon: float
    tau: float
    lambda_: float
    temperature: float
    evidence_metric: str
    cost_metric: str
    scheme: str

    def __post_init__(self):
        for field in ("epsilon", "tau", "lambda_", "temperature"):
            require(
                isinstance(getattr(self, field), (int, float)) and not isinstance(getattr(self, field), bool),
                "Parameter must be a JSON number: " + field,
            )
        number(self.epsilon, "epsilon", 0, 1)
        number(self.tau, "tau", 0, 1)
        number(self.lambda_, "lambda", 0)
        number(self.temperature, "temperature", 0)
        require(self.temperature > 0, "temperature must be positive")
        require(self.scheme in SCHEMES, "Unknown scheme")
        require(self.evidence_metric in EVIDENCE_METRICS, "Unsupported evidence_metric")
        require(self.cost_metric in COST_METRICS, "Unsupported cost_metric")

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["lambda_"] = data.pop("lambda")
        return cls(**data)

    def to_dict(self):
        data = asdict(self)
        data["lambda"] = data.pop("lambda_")
        return data

@dataclass(frozen=True)
class Question:
    question_id: str
    type: str
    question_text: str
    gold_answer: str
    ks: tuple
    f1: tuple
    em: tuple
    tokens: dict
    evidence: tuple
    predictions: tuple
    original_hard_oracle_k: int

def normalize_cost(tokens):
    values = tuple(number(v, "cost tokens", 0) for v in tokens)
    require(values and max(values) > 0, "Cost normalization requires a positive all-k maximum")
    maximum = max(values)
    return tuple(v / maximum for v in values)

def stable_softmax(utilities, mask, temperature):
    require(len(utilities) == len(mask) and any(mask), "Softmax needs a nonempty aligned candidate set")
    number(temperature, "temperature", 0)
    require(temperature > 0, "temperature must be positive")
    for u in utilities:
        number(u, "utility")
    maximum = max(u for u, valid in zip(utilities, mask) if valid)
    weights = [math.exp((u - maximum) / temperature) if valid else 0.0 for u, valid in zip(utilities, mask)]
    total = math.fsum(weights)
    return tuple(w / total for w in weights)

def build_label(question, params):
    f1, ks = question.f1, question.ks
    best = max(f1)  # Always all k, before any evidence restriction.
    ev = tuple(row[params.evidence_metric] for row in question.evidence)
    costs = normalize_cost(question.tokens[params.cost_metric])  # Always all k.
    f1_pass = tuple(v + BOUNDARY_ATOL >= best - params.epsilon for v in f1)
    evidence_pass = tuple(v + BOUNDARY_ATOL >= params.tau for v in ev)
    use_f1, use_ev = SCHEMES[params.scheme]
    mask = tuple((not use_f1 or a) and (not use_ev or b) for a, b in zip(f1_pass, evidence_pass))
    utility = tuple(f - params.lambda_ * c for f, c in zip(f1, costs))
    count = sum(mask)
    status = "all_zero_f1" if best == 0 else "empty_candidates" if count == 0 else "valid"
    probabilities, argmax_k, entropy, normalized_candidate = None, None, None, None
    if status == "valid":
        probabilities = stable_softmax(utility, mask, params.temperature)
        # Utility ordering avoids temperature/underflow-induced argmax changes. Ties choose smallest k.
        index = max((i for i, yes in enumerate(mask) if yes), key=lambda i: utility[i])
        argmax_k = ks[index]
        entropy = -math.fsum(p * math.log(p) for p in probabilities if p > 0)
        normalized_candidate = entropy / math.log(count) if count > 1 else None
    return {
        "question_id": question.question_id,
        "type": question.type,
        "parameters": params.to_dict(),
        "max_f1": best,
        "f1_pass": f1_pass,
        "evidence_pass": evidence_pass,
        "candidate_mask": mask,
        "valid_candidates": tuple(k for k, yes in zip(ks, mask) if yes),
        "valid_candidate_count": count,
        "normalized_cost_curve": costs,
        "utility_curve": utility,
        "soft_label": probabilities,
        "label_status": status,
        "argmax_soft_label_k": argmax_k,
        "original_hard_oracle_k": question.original_hard_oracle_k,
        "entropy": entropy,
        "entropy_normalized_all_k": entropy / math.log(len(ks)) if entropy is not None and len(ks) > 1 else None,
        "entropy_normalized_candidates": normalized_candidate,
        "has_unknown_sentence_evidence": any(r["sentence_unknown"] for r in question.evidence),
    }
