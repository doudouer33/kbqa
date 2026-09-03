# ONER input reconstruction

`reconstruct_oner.py` reconstructs the retrieval context and complete model
prompt for one ONER prediction directory. It replays retrieval only; it never
calls the language model and does not change the original prediction files.

For each directory, it performs the first reconstruction stage in five parts:

1. reads the saved Jsonnet config, variable replacements, evaluation input,
   prediction JSON, and chain file;
2. rebuilds the original ONER retrieval query;
3. replays BM25 retrieval against the configured corpus;
4. checks the evaluation question and the retrieved title order against the
   saved chain;
5. follows the current `run.py` execution path to rebuild the formatted context,
   fitted prompt, and one-user-message request, then writes them to JSON.

Run it from the repository's Python environment (the same environment used by
`run.py`):

```bash
python reconstruction/reconstruct_oner.py \
  predictions/dev_500/oner_qa_gpt_nq____prompt_set_1___bm25_retrieval_count__2___distractor_count__1 \
  --retriever-port 8000
```

The default output is written into the supplied experiment directory as:

```text
reconstructed_inputs__<dataset-and-split>.json
```

Use `--limit 5` for a small trial. Limited runs get a separate
`__limit_5.json` suffix. Existing outputs are not overwritten unless `--force`
is passed.

## Validation status

- `replayed_match`: the evaluation question agrees with the chain and the
  replayed title sequence exactly matches the saved title sequence. This is
  suitable for reconstructed token accounting, while acknowledging that old
  ONER chains did not save document IDs.
- `failed`: retrieval, rendering, question validation, or title/order
  validation failed. Such examples have `valid_for_token_count: false` and
  should not be included silently in an aggregate.

The output includes, for every question:

- the query after the original `remove_wh_words` processing;
- retrieved IDs, titles, paragraph text, and scores;
- the context after the original 350-word clipping and title formatting;
- the complete prompt after the original prompt-fitting step;
- the effective prompt-fitting settings and single-user-message request;
- chain, question, title-order, and prediction-answer validation results.

Token counting is intentionally a separate step. The reconstruction file keeps
the full prompt and model name so that GPT and Qwen tokenizers can be applied
with different chat-template rules later.

## Batch run: dev_500 ONER GPT NQ

To reconstruct BM25 retrieval counts 1 through 15 sequentially against the
retriever on port 8000:

```bash
bash reconstruction/run_oner_dev500_nq_counts_1_to_15.sh
```

The launcher uses eight concurrent retrieval requests inside each experiment,
runs strict chain validation, and overwrites only the derived reconstruction
JSON files. Its Python executable, retriever address, and concurrency can be
overridden with `KBQA_RECON_PYTHON`, `KBQA_RETRIEVER_HOST`,
`KBQA_RETRIEVER_PORT`, and `KBQA_RECON_WORKERS`.

## GPT input-token counting

`count_input_tokens.py` reads the reconstructed `request_messages`, skips every
item whose `valid_for_token_count` is false, and writes this file into the same
experiment directory:

```text
input_token_counts__<dataset-and-split>.json
```

The `token_counts` object maps each counted question ID to its estimated Chat
Completions input-token count. The JSON also records skipped IDs, summary
statistics, model, encoding, framing constants, and the reconstruction hash.
Because the historical API response usage was not saved, this is explicitly an
offline `tiktoken` estimate rather than an original server-reported count.

Run one directory:

```bash
python reconstruction/count_input_tokens.py \
  predictions/dev_500/oner_qa_gpt_nq____prompt_set_1___bm25_retrieval_count__2___distractor_count__1
```

Run BM25 retrieval counts 1 through 15:

```bash
bash reconstruction/run_oner_dev500_nq_token_counts_1_to_15.sh
```
