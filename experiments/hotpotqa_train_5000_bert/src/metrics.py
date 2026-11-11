"""Copied pure functions: official HotpotQA Answer F1 and established sentence evidence matching.
Sources and exact source hashes: ../reports/algorithm_sources.json.
No runtime import of historical experiments. Evidence uses literal sentence coverage,
with uncertain matches retained as unknown and excluded from lower recall.
"""
import re, string, html, unicodedata
from collections import Counter
MAX_WORDS = 350

def require(condition, message):
    if not condition: raise ValueError(message)

def index_rows(rows, field, source):
    result = {}
    for row in rows:
        key = row[field]
        require(key not in result, 'Duplicate key in '+str(source))
        result[key] = row
    return result

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def norm(text):
    # Preserve punctuation, digits and accents: title punctuation can identify
    # different entities (Money, Money, Money vs Money! Money! Money!).
    text = unicodedata.normalize("NFKC", html.unescape(text)).casefold()
    return " ".join(text.split())

def compact(text):
    # Only whitespace is removed. No fuzzy/token-overlap matching is used.
    return "".join(norm(text).split())

def loose_title(text):
    # Used to mark uncertainty, NEVER to award a match.
    return "".join(c for c in norm(text) if c.isalnum())

def render_context(doc):
    body = " ".join(doc["paragraph_text"].split(" ")[:MAX_WORDS]).strip()
    return body if body.startswith("Wikipedia Title: ") else "Wikipedia Title: " + doc["title"] + "\n" + body

def model_doc(doc):
    text = render_context(doc)
    # The added title is not counted as a supporting sentence.
    body = text.split("\n", 1)[1] if text.startswith("Wikipedia Title: ") and "\n" in text else ""
    return {"title": doc["title"], "paragraph_text": body}

def coverage_state(hit, unknown, total):
    require(total > 0 and 0 <= hit <= total and 0 <= unknown <= total - hit, "Bad evidence counts")
    if hit == total:
        return "complete"
    if unknown:
        return "unknown"
    return "partial" if hit else "zero"

def prepare_gold(sample, raw, unknown_facts=()):
    qid = sample["question_id"]
    for key, rawkey in (("question_text", "question"), ("type", "type"), ("level", "level")):
        require(sample[key] == raw[rawkey], f"Raw/processed {key} mismatch {qid}")
    answers = [a for obj in sample["answers_objects"] for a in obj["spans"]]
    require(answers == [raw["answer"]], f"Gold answer mismatch {qid}")
    raw_context = index_rows([{"title": t, "sentences": ss} for t, ss in raw["context"]], "title", qid)
    facts = [tuple(f) for f in raw["supporting_facts"]]
    require(len(facts) == len(set(facts)), f"Duplicate supporting fact {qid}")
    require(facts, f"No supporting facts {qid}")
    processed = index_rows(sample["contexts"], "title", qid)
    require(set(processed) == set(raw_context), f"Raw/processed titles differ {qid}")
    titles = {f[0] for f in facts}
    for title, row in processed.items():
        require(isinstance(row["is_supporting"], bool), f"Invalid is_supporting {qid}")
        require(row["is_supporting"] == (title in titles), f"Support flag mismatch {qid}")
        require(compact(row["paragraph_text"]) == compact("".join(raw_context[title]["sentences"])),
                f"Raw/processed text differs {qid}/{title}")
    paragraphs = []
    for title in sorted(titles):
        require(title in raw_context, f"Fact title absent from raw context {qid}/{title}")
        sentences = raw_context[title]["sentences"]
        indices = [i for t, i in facts if t == title]
        for i in indices:
            valid = isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(sentences)
            if not valid:
                require((title, i) in unknown_facts, f"Bad fact index {qid}/{title}/{i}")
            else:
                require(bool(sentences[i].strip()), f"Empty support sentence {qid}/{title}/{i}")
        text = "".join(sentences).strip()
        paragraphs.append({"title": title, "text": text, "norm": norm(text), "compact": compact(text),
                           "facts": [{"sentence_index": i, "text": sentences[i].strip() if 0 <= i < len(sentences) else None,
                                      "unavailable_official_sentence": not 0 <= i < len(sentences)} for i in indices]})
    return paragraphs

def match_evidence(gold, documents):
    """Conservative literal coverage, with explicit per-item lower/upper bounds.

    Paragraph identity requires exact normalized text or a substantial same-title
    prefix. A title-only/text-version conflict is unknown. Sentence occurrence is
    checked separately in the supplied (already rendered) text. Long exact
    cross-title occurrences count as sentence evidence but not as paragraph hits.
    """
    docs = [{"title": d["title"], "n": norm(d["paragraph_text"]), "c": compact(d["paragraph_text"])} for d in documents]
    paragraphs, facts = [], []
    for g in gold:
        methods, candidates = [], []
        full_text = False
        for j, d in enumerate(docs):
            same = norm(d["title"]) == norm(g["title"])
            near = loose_title(d["title"]) == loose_title(g["title"])
            exact = d["n"] == g["norm"]
            spacing = d["c"] == g["compact"]
            if (exact or spacing) and (same or len(g["compact"]) >= 80):
                methods.append((j, "exact" if exact else "whitespace_only"))
                full_text = True
            elif same and len(d["c"]) >= 80 and len(d["n"].split()) >= 12 and g["compact"].startswith(d["c"]):
                methods.append((j, "same_title_prefix"))
            elif same or near:
                candidates.append(j)
        status = "present" if methods else "unknown" if candidates else "absent"
        paragraphs.append({"title": g["title"], "status": status, "full_text_present": full_text,
                           "matches": [{"document_index": j, "method": m} for j, m in methods],
                           "ambiguous_document_indices": candidates})
        for fact in g["facts"]:
            # User-approved extension for the two recorded out-of-range Train
            # facts: retain denominator, always unknown, never count as a hit.
            if fact.get("unavailable_official_sentence", False):
                facts.append({"title": g["title"], "sentence_index": fact["sentence_index"],
                              "status": "unknown", "matches": [], "reason": "unavailable_official_sentence"})
                continue
            fn, fc = norm(fact["text"]), compact(fact["text"])
            matches, short_unanchored = [], False
            for j, d in enumerate(docs):
                same = norm(d["title"]) == norm(g["title"])
                exact, spacing = fn in d["n"], fc in d["c"]
                if not (exact or spacing):
                    continue
                anchored = same or any(j == index for index, _ in methods)
                long_literal = len(fc) >= 40 and len(fn.split()) >= 6
                if anchored or long_literal:
                    matches.append({"document_index": j, "method": "exact" if exact else "whitespace_only",
                                    "cross_title": not same})
                else:
                    short_unanchored = True
            fs = "present" if matches else "unknown" if status == "unknown" or short_unanchored else "absent"
            facts.append({"title": g["title"], "sentence_index": fact["sentence_index"], "status": fs, "matches": matches})
    result = {"paragraphs": paragraphs, "facts": facts}
    for unit, rows in (("paragraph", paragraphs), ("sentence", facts)):
        hit = sum(x["status"] == "present" for x in rows)
        unknown = sum(x["status"] == "unknown" for x in rows)
        result.update({unit + "_hit": hit, unit + "_unknown": unknown, unit + "_total": len(rows),
                       unit + "_status": coverage_state(hit, unknown, len(rows)),
                       unit + "_recall_lower": hit / len(rows),
                       unit + "_recall_upper": (hit + unknown) / len(rows)})
    result["full_paragraph_text_count"] = sum(x["full_text_present"] for x in paragraphs)
    return result
