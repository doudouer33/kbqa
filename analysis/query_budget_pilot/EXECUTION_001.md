# 2026-10-30 实际执行记录

用户在准备阶段完成后明确授权直接连接 SSH 别名 `a800`、使用第一张显卡并继续操作。本次已经完成真实 Embedding 提取、SCP 回传、18 组 CPU 训练、逐题离线评价及汇总。下述结果来自真实 BGE-M3 向量；准备阶段的合成数据测试另有独立记录。

## A800 执行与文件位置

- SSH 别名：`a800`；物理 GPU：**0，NVIDIA A800-SXM4-80GB**。
- GPU UUID：`GPU-1173bf72-51fb-2860-8f92-23860ba176f5`。
- 上传的独立任务：`/nas/luoyichen/query_budget_pilot/a800_embedding_job/`，未克隆仓库，未上传答案或标签。
- 模型目录：`/nas/luoyichen/models/bge-m3/`。
- 提取环境：`/nas/luoyichen/envs/query-budget-embedding/`。
- 下载/运行缓存：`/nas/luoyichen/cache/`；临时文件：`/nas/luoyichen/tmp/`。
- 真实输出：`/nas/luoyichen/query_budget_pilot/results/run_001/embeddings.npz`、`embedding_metadata.json`、`verification.json`。
- 远端日志：`/nas/luoyichen/query_budget_pilot/logs/`。

新建 venv 通过 `--system-site-packages` 只读复用现有环境 `/nas/luoyichen/miniconda3/envs/vllm061/` 中的 CUDA PyTorch，原环境未修改；没有调用 vLLM。实际 Python 3.10.20、numpy 1.26.4、torch 2.4.0+cu118、Sentence Transformers 3.4.1、Transformers 4.45.1、huggingface-hub 0.36.2，`pip check` 通过。

A800 直连 Hugging Face 超时，使用 `hf-mirror.com` 下载固定版本：`5617a9f61b028005a4858fdac845db406aefb181`。全部 10 个必要文件均对照 [Hugging Face 官方文件元数据](https://huggingface.co/api/models/BAAI/bge-m3?blobs=true) 校验了 Git Blob SHA1 或 LFS SHA256；没有下载 ONNX、Sparse 或 ColBERT 权重。`pytorch_model.bin` SHA256 为 `b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38`。模型目录的 `model_provenance.json` 保存逐文件摘要。

最初的 PyPI 下载速度较慢，仅终止了本次任务自己的 pip 下载进程，随后通过清华镜像完成依赖安装。安装、模型下载和提取命令留存在本地 `outputs/execution_001/remote_*.sh`；日志已传回 `outputs/execution_001/a800_logs/`。

提取时设置 `CUDA_VISIBLE_DEVICES=0`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，模型从已验证的本地目录加载：

```sh
cd /nas/luoyichen/query_budget_pilot/a800_embedding_job
. ./a800_env.sh
. /nas/luoyichen/envs/query-budget-embedding/bin/activate
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python extract_embeddings.py \
  --input /nas/luoyichen/query_budget_pilot/a800_embedding_job/queries.jsonl \
  --output /nas/luoyichen/query_budget_pilot/results/run_001/embeddings.npz \
  --model /nas/luoyichen/models/bge-m3 \
  --revision 5617a9f61b028005a4858fdac845db406aefb181 \
  --local-files-only --batch-size 32 --device cuda:0
```

该命令是本次实际参数的记录，既有输出会拒绝覆盖。重跑请改用新结果目录。

模型冻结，`eval()` 与 `inference_mode()`，float32 前向，CLS Dense Embedding、L2 归一化。最大实际 Query 长度为 58 token，没有截断。1000 个 ID、1024 维、float32、全部有限值、单位范数、输入与输出 SHA256 均在 A800 和回传后分别校验通过。提取进程退出后，第 0 张 GPU 显存占用恢复为 0 MiB。

Embedding SHA256：`bd76637583caa1ed84ba662cbca2b430171c0c1815fb3c53321b0af9a0796e28`。

Query SHA256：`876d1ea0f5ad699a793a84ced0be45bd0a80c8277241140b2a64f86217d91304`。

## CPU 正式训练

回传文件保存于 `analysis/query_budget_pilot/inputs/`。模型权重、Embedding、逐次训练中间产物和运行缓存均被 Git 忽略；最终报告、JSON/CSV 汇总和 PNG 图四个产物纳入 Git。4090 服务器的 GPU 对训练进程不可见，PyTorch 线程数为 2。

```sh
cd /media/hnu/hnu2021/dengxin/luoyichen/kbqa
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  /home/dengxin/miniconda3/envs/adaptiverag/bin/python -u \
  analysis/query_budget_pilot/run_matrix.py \
  --config analysis/query_budget_pilot/config.json \
  --device cpu --num-threads 2 \
  --output analysis/query_budget_pilot/outputs/matrix_001
```

执行了预先固定的三个目标、两种架构、种子 13/42/2026，共 18 组。没有依据 Test 结果修改参数或重新选择检查点。各 seed 内共用 700/100/200 划分；每个完整测试集包含 150 个有效标签样本和 **50 个无标签样本**。训练损失仅消费 Train 的 526 个有效样本，早停仅消费 Validation 的 76 个有效样本。

## 完整测试集结果

下表为三个种子的均值 ± 样本标准差。种子同时改变划分与初始化；跨种子测试集可以重叠，这些波动不是独立置信区间。所有测试集都包含无标签题。

| 方法 | Answer F1 | Context Tokens | Evidence Recall | k=1 比例 |
| --- | --- | --- | --- | --- |
| Fixed k=1 | 0.5036 ± 0.0289 | 94.08 ± 3.19 | 0.3814 ± 0.0135 | 100.0% |
| Fixed k=3 | 0.5282 ± 0.0250 | 254.18 ± 4.32 | 0.5987 ± 0.0292 | 0.0% |
| Fixed k=5 | 0.5680 ± 0.0176 | 405.21 ± 4.60 | 0.6470 ± 0.0220 | 0.0% |
| Fixed k=13 | 0.5959 ± 0.0240 | 999.95 ± 18.06 | 0.7281 ± 0.0128 | 0.0% |
| Linear / Minimum Feasible | 0.5036 ± 0.0289 | 94.08 ± 3.19 | 0.3814 ± 0.0135 | 100.0% |
| Linear / Original Soft | 0.5277 ± 0.0282 | 163.51 ± 42.59 | 0.4823 ± 0.0468 | 51.3% |
| Linear / Sharper Soft | 0.5058 ± 0.0357 | 100.54 ± 12.05 | 0.3948 ± 0.0292 | 92.5% |
| MLP / Minimum Feasible | 0.5036 ± 0.0289 | 94.08 ± 3.19 | 0.3814 ± 0.0135 | 100.0% |
| MLP / Original Soft | 0.5352 ± 0.0247 | 141.89 ± 4.04 | 0.4725 ± 0.0147 | 43.0% |
| MLP / Sharper Soft | 0.5036 ± 0.0289 | 94.29 ± 3.27 | 0.3822 ± 0.0140 | 99.5% |

Original Soft + MLP 相对 Fixed k=3 的同 seed、同测试集配对差：

- F1 增加 **0.0070 ± 0.0029**，即 **0.70 ± 0.29 个百分点**。
- Context Tokens 减少 **112.29 ± 7.42**；两者均值之比对应 **44.18%** 的 Context 节省。
- Prompt Tokens 从 6672.26 降至 6559.96，对应 **1.68%** 节省。44.18% 不能解释为完整 API Prompt 成本节省。
- Evidence Recall 降低 **0.1262 ± 0.0275**，即 **12.62 ± 2.75 个百分点**。

本轮在预设的 F1–Context 比较上观察到有希望的权衡，但证据覆盖明显下降。Minimum Feasible 两种架构在全部三个种子的测试集上均预测 k=1；Sharper Soft 也高度集中于 k=1。Original Soft 两种架构未触发 95% 同类的测试集塌缩标志，但仍需结合预算分布解释。

这些是同一批已用于方法设计的 HotpotQA Dev 上的探索性结果，不能当作独立泛化证明。离线 Max-F1 Oracle 仅是读取真实逐 k 分数的参考上界，不能在线部署；完整报告中已单独标注。

## 产物与核验

- 完整报告：`outputs/matrix_001/report/report.md`。
- 全部划分、题型、有效状态、种子汇总：`outputs/matrix_001/report/summary.json`、`summary.csv`。
- F1–Token 图：`outputs/matrix_001/report/f1_token.png`。
- 逐次训练：`outputs/matrix_001/seed_*/{model}_{label}/training.json`、`history.json`、`best.pt`。
- 每次模型的全 1000 题概率和预算：对应 `evaluation/predictions.jsonl`；逐 k 查表指标在 `evaluation/metrics.json`。
- 真实数据独立核验：`outputs/execution_001/production_validation.json`。
- 可提交的小型执行审计：`verification/execution_001.json`。

独立核验检查了 18,000 条预测的 ID、概率和 argmax，重新计算了 11,664 个离线指标标量，确认各 seed 的所有模型测试 ID 相同、每个测试集保留 50 道无标签题、检查点均在 CPU 训练。原有 RAG、Soft Oracle 和诊断文件保持不变。
