# HotpotQA Train5000: Original Soft Labels + BERT end-to-end

Execution completed successfully. Results are in [`reports/report.md`](reports/report.md); the selected checkpoint is epoch 2 of 5, trained on the pinned RTX 4090. The primary deployment comparison is Answer F1–Context Tokens against fixed-k baselines; Train Mean remains an auxiliary label-learning diagnostic.

This independent experiment builds offline labels from the completed 5000×15 formal RAG run, then fully fine-tunes BERT on **question text only**. It never imports a historical experiment module, calls Qwen/BM25, or modifies the formal RAG results.

The approved handling of two invalid original supporting-fact sentence indices is `unknown` with the **original denominator retained**. Only the two exact question/title/index triples in `config/experiment.json` are allowed; unexpected invalid annotations still fail. These sentences cannot be counted as evidence hits. No paragraph-recall substitution or one-hot repair is applied.

## Fixed configuration

- k=1..15, Scheme D, epsilon=0.1, tau=0.5, lambda=0.1, T=0.1.
- Answer F1: copied official HotpotQA functions, including yes/no/noanswer rules.
- Context cost: Qwen tokenizer on actual saved `effective_context_text`, excluding few-shot examples, question, generation and chat-template overhead. Missing API usage is never fabricated.
- Evidence: original sentence-level literal matcher with lower/upper/unknown counts. Lower recall = confirmed sentence hits / original supporting-fact count.
- Cost normalization uses each question's maximum over **all 15 k**. Candidates satisfy both F1 and evidence thresholds; probabilities are masked softmax of `(F1 - lambda * normalized_cost) / T`.
- Invalid labels retain `all_zero_f1` / `empty_candidates`, with null probabilities.
- BERT: `bert-base-uncased`, all parameters trainable plus a 15-class head.
- Seed 42; all sample IDs shuffled independently of labels into 3500/500/1000 Train/Validation/Test. Only valid labels train or contribute Soft CE/KL; answer/cost evaluation uses all common test IDs, with a separate valid-only view.
- AdamW, lr 2e-5, batch 16, five epochs, fp16, question max length 128. Best checkpoint chosen by minimum Validation Soft CE.
- Required GPU UUID and name are pinned in configuration. The launcher checks identity, CUDA computation and memory, and fails instead of falling back to another GPU or CPU.

## Files

```text
config/experiment.json           label, training, split, annotation and GPU rules
references/dev1000_original_soft_labels.jsonl
                                copied historical reference DATA, not imported code
src/metrics.py                  official Answer F1 + copied sentence matcher
src/soft_oracle.py              copied Original Soft Oracle pure functions
src/build_data.py               read-only offline curves, labels and integrated checks
src/device.py                   RTX 4090 identity and resource checks
src/train.py                    full BERT training and best-checkpoint inference
src/evaluate.py                 same-test baselines, CE/KL, F1/tokens, specificity
src/run_pipeline.py             single-instance durable stage runner
tests/                          algorithm, missing-fact and evaluation regressions
data/queries.jsonl              ONLY question ID and question text
data/curves.jsonl               F1/tokens/sentence evidence curves for all 5000×15
data/labels.jsonl               candidate masks, probabilities and original statuses
data/splits.json               immutable split IDs and valid-label counts
data/metadata.json             input/output/code fingerprints and construction checks
data/train_mean_distribution.json
                                distribution fitted ONLY on training valid labels
data/bert_predictions.jsonl     selected checkpoint's Validation/Test distributions
checkpoints/best/               BERT weights, tokenizer/config and selection metadata
logs/                          data build, batches, epochs and pipeline logs
reports/label_distribution.json  status/probability/entropy/k and Dev1000 comparison
reports/gpu_preflight.json       GPU UUID/name/CUDA/resources before training
reports/training_summary.json   epoch history, selected epoch and encoder changes
reports/evaluation.json         full metrics, paired CIs and permutation diagnostic
reports/report.md               human-readable results and plots
reports/status.json             current execution status
```

Algorithms, source hashes and the copied 1000-question reference data provenance are recorded in `reports/algorithm_sources.json`. The builder reproduces all 1000 historical labels, candidates and argmax choices to 1e-12 probability tolerance using **only local implementations and the copied reference data**. The historical Dev1000 split was balanced by question type; the natural Train5000 sample differs in population, so distribution comparisons are descriptive.

The existing preprocessing deduplicates equal paragraph bodies, retains their last title, and clips stored paragraphs at 1000 space-separated words. The Train join reproduces those transformations to validate alignment. Sentence evidence always uses original raw supporting sentences and actual saved reader contexts.

## Execute

From repository root, using the existing environment and locally cached tokenizers/weights:

```sh
PYTHON=/home/dengxin/miniconda3/envs/adaptiverag/bin/python
"$PYTHON" -m unittest discover -s experiments/hotpotqa_train_5000_bert/tests -v
"$PYTHON" -u experiments/hotpotqa_train_5000_bert/src/run_pipeline.py
```

Do not run the pipeline twice concurrently or overwrite existing prepared data/checkpoints. The completed durable execution used tmux session `hotpot5000-bert-4090`; inspect `reports/status.json` and `logs/pipeline.log` for progress.

Evaluation compares BERT, the Train Mean Distribution, all Fixed-k policies and the offline max-F1 oracle on identical test IDs. Expected policy metrics and deterministic argmax metrics are distinct. Exact CE/KL can be infinite for deterministic policies incompatible with soft targets; separately named epsilon-floor metrics do not replace exact values. A valid-only Original Soft Oracle reference is also reported. Validation alone chooses checkpoints and the displayed best-fixed-k comparison. Test metrics and permutation/bootstrap diagnostics are descriptive, never model-selection criteria.

## GitHub 上的文件

仓库包含源码、固定配置、报告与图表、训练 batch/epoch 日志、划分 ID、完整 Validation/Test 预测、测试决策、Mean Distribution，以及最佳模型的配置和选择记录。

[`examples/`](examples/README.md) 提供 7 道真实题目的输入、标签和曲线示例。完整 5000 道输入/标签/曲线与约 418 MiB 模型权重保留在本地，未纳入普通 Git 历史；文件指纹保存在报告中。`references/` 中历史 1000 道标签参考数据已复制到本目录，回归测试不需要导入或读取旧实验目录。

结果解读以固定 k 为主：相对 k=15，BERT 节省87.2%上下文tokens，F1下降4.48个百分点；相对 k=2，节省18.9%tokens，F1下降1.05个百分点。Train Mean 与 Soft CE/KL 为辅助标签学习诊断。当前结果体现成本和质量的交换，不能声称在同等成本下已经全面优于固定 k。
