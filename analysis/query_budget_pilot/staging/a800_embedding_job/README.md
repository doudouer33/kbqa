# A800 独立 BGE-M3 Query 提取

本目录不需要 kbqa 仓库，也不包含答案、标签、逐 k 结果或索引。所有 A800 新建的代码、环境、模型、缓存与输出均放在 `/nas/luoyichen/` 下。此说明中的命令由用户手动执行。

目录位置：`/nas/luoyichen/query_budget_pilot/a800_embedding_job`。

## 环境与输入

建议 Python 3.10/3.11；先确认 `python3 --version`。若系统 Python 版本不符，使用已有的 3.10/3.11 Python 可执行文件替代下面的 `python3`。

```sh
mkdir -p /nas/luoyichen/query_budget_pilot /nas/luoyichen/envs
cd /nas/luoyichen/query_budget_pilot/a800_embedding_job
. ./a800_env.sh
python3 -m venv /nas/luoyichen/envs/query-budget-embedding
. /nas/luoyichen/envs/query-budget-embedding/bin/activate
python -m pip install --upgrade pip
# CUDA 12.1 wheel；若服务器已有可用 CUDA PyTorch，可复用 /nas/luoyichen 下的环境，跳过此行。
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-embedding.txt
python -m pip check
python -c 'import torch, numpy, sentence_transformers; print(torch.__version__, numpy.__version__, sentence_transformers.__version__)'
python - <<'PY'
import hashlib, json
from pathlib import Path
m = json.loads(Path('job_manifest.json').read_text())
for name, digest in m['files_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
rows = [json.loads(s) for s in Path('queries.jsonl').read_text().splitlines()]
assert len(rows) == len({r['question_id'] for r in rows}) == 1000
assert all(set(r) == {'question_id', 'question_text'} and r['question_text'].strip() for r in rows)
print('Job hashes and 1000 unique queries OK; source SHA256:', m['source_dataset_sha256'])
PY
nvidia-smi
```

## 提取（以下两种方式只选一种）

将 `<GPU_INDEX>` 替换为用户确认空闲的物理卡号。`CUDA_VISIBLE_DEVICES` 只暴露这一张卡，进程内用 `cuda:0`。不启动多卡、不训练模型。输出目录已存在文件会拒绝覆盖；重跑时换 `run_002`。

默认模型 ID 会复用 `/nas/luoyichen` 下设置的缓存，缺少文件时下载到这些路径：

```sh
cd /nas/luoyichen/query_budget_pilot/a800_embedding_job
. ./a800_env.sh
. /nas/luoyichen/envs/query-budget-embedding/bin/activate
GPU_INDEX='<GPU_INDEX>'
mkdir -p /nas/luoyichen/query_budget_pilot/results/run_001
CUDA_VISIBLE_DEVICES="$GPU_INDEX" python extract_embeddings.py \
  --input /nas/luoyichen/query_budget_pilot/a800_embedding_job/queries.jsonl \
  --output /nas/luoyichen/query_budget_pilot/results/run_001/embeddings.npz \
  --model BAAI/bge-m3 --batch-size 32 --device cuda:0 \
  --cache-folder /nas/luoyichen/models/sentence-transformers
```

已有本地模型时，用实际的 `/nas/luoyichen/…` 模型目录替换示例（必须是完整、兼容 Sentence Transformers 的 BGE-M3 目录，包含 CLS pooling 配置）：

```sh
CUDA_VISIBLE_DEVICES="$GPU_INDEX" HF_HUB_OFFLINE=1 python extract_embeddings.py \
  --input /nas/luoyichen/query_budget_pilot/a800_embedding_job/queries.jsonl \
  --output /nas/luoyichen/query_budget_pilot/results/run_001/embeddings.npz \
  --model /nas/luoyichen/models/bge-m3 --local-files-only \
  --batch-size 32 --device cuda:0
```

可加 `--revision <MODEL_COMMIT>` 固定 Hub 版本。默认 float32 前向、输出 float32、L2 归一化、无额外 Query 指令、最长 8192 token；遇到将被截断的 Query 会报错。提取包含 `eval()`、冻结参数和 `inference_mode()`，只输出 Dense Embedding。

## 不加载模型的输出检查

```sh
python extract_embeddings.py \
  --input /nas/luoyichen/query_budget_pilot/a800_embedding_job/queries.jsonl \
  --output /nas/luoyichen/query_budget_pilot/results/run_001/embeddings.npz \
  --verify-only
```

应显示 `shape: [1000, 1024]`、`question_id_count: 1000` 与完整元数据；同时验证有限值、float32、ID 集、L2 范数、输入及 NPZ SHA256。该检查无需 GPU，不加载模型。

传回 4090 的文件仅为：

- `/nas/luoyichen/query_budget_pilot/results/run_001/embeddings.npz`
- `/nas/luoyichen/query_budget_pilot/results/run_001/embedding_metadata.json`

传输命令和 Windows PowerShell 中转方案见 4090 仓库 `analysis/query_budget_pilot/README.md`。

API 依据：[BGE-M3 官方模型说明](https://huggingface.co/BAAI/bge-m3)、[SentenceTransformer 文档](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html)。提取依赖固定在 `sentence-transformers==3.4.1`；实际软件版本和可取得的模型 commit 写入元数据。
