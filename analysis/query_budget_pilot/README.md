# Query-only Dynamic Top-K 小规模训练准备

本目录新增独立实验，仅读取已有 1000 题、k=1…15 的正式离线统计。**2026-10-30 已按用户后续授权，通过 SSH 别名 `a800` 在物理 GPU 0 完成真实 BGE-M3 Embedding 提取、校验与回传，并在 4090 服务器 CPU 完成 18 组正式训练及离线评价。** 执行记录和主要结果见 [EXECUTION_001.md](EXECUTION_001.md)，完整报告为 `outputs/matrix_001/report/report.md`。没有重新运行 RAG、检索或 LLM。

原始“准备上传”阶段审计保留在 `verification/delivery_audit.json`；实际执行审计为 `verification/execution_001.json`。下面的手动命令保留供复现，当前无需再次手动上传或提取；重跑时应使用新的结果目录。

所有 A800 新建文件统一放在 `/nas/luoyichen/`：代码 `/nas/luoyichen/query_budget_pilot/`，模型 `/nas/luoyichen/models/`，缓存 `/nas/luoyichen/cache/`，环境 `/nas/luoyichen/envs/`，临时文件 `/nas/luoyichen/tmp/`。`a800_env.sh` 显式配置这些位置。

## Original Soft Label 可学习性离线诊断

新增 `diagnose_soft_labels.py`，只读取保存的标签、划分、训练元数据与六份 Original Soft Label `predictions.jsonl`。每个 seed 的 Mean Distribution Baseline 只使用有效 Train 标签；所有方法在相同的有效 Test 问题上比较。包含 KL / Soft CE、逐题胜率、预测分布变化、大 k 质量与 Pearson / Spearman，以及消除逐题变化后的概率对照。

已完成结果见 [diagnostics/original_soft_001/report.md](diagnostics/original_soft_001/report.md)。脚本仅依赖 NumPy / Matplotlib，不导入训练、Embedding 或 RAG 模块；不会读取权重或向量。输入缺失时报告缺失路径并退出（退出码 2），不会自动补训。结果目录必须是新目录，并位于既有训练/准备结果之外；复现时可运行：

```sh
/home/dengxin/miniconda3/envs/adaptiverag/bin/python analysis/query_budget_pilot/diagnose_soft_labels.py \
  --output analysis/query_budget_pilot/diagnostics/original_soft_002
/home/dengxin/miniconda3/envs/adaptiverag/bin/python -m unittest discover \
  -s analysis/query_budget_pilot/tests -p test_soft_label_diagnostic.py -v
```

这组新增测试只验证计算与文件读取，不调用训练或生成 Embedding。诊断产物与原 `outputs/matrix_001/` 分开保存。

## 文件与已准备的数据

```text
analysis/query_budget_pilot/
  common.py                    复用 Soft Oracle、严格输入校验、分层 ID 划分
  prepare_queries.py           仅导出 ID 和原始 Query 文本
  prepare_labels.py            三种标签、已有逐 k 曲线、三组划分
  extract_embeddings.py        自包含提取/校验脚本，无仓库依赖
  pack_a800_job.py              创建最小上传目录，拒绝覆盖
  train.py                     CPU Linear/MLP，Soft CE、AdamW、早停
  evaluate.py                  完整测试集离线评价及跨 seed 报告
  run_matrix.py                顺序执行 18 次 CPU 实验并自动生成报告
  smoke_test.py                仅合成数据端到端检查；临时模型/向量会删除
  config.json
  requirements-training.txt
  requirements-embedding.txt
  ruff.toml                    独立 lint 配置，兼容现有仓库旧配置
  a800_env.sh
  A800_README.md                上传包说明的源文件
  tests/synthetic.py
  tests/test_pilot.py
  data/prepared_001/
    queries.jsonl
    query_metadata.json
    labels.jsonl
    offline_curves.jsonl
    splits/seed_13.json
    splits/seed_42.json
    splits/seed_2026.json
    metadata.json
    validation.json
  staging/a800_embedding_job/   下节列出全部 6 个文件
  inputs/                      等待回传 NPZ 和 embedding_metadata.json
  outputs/                     训练/评价输出；仅 matrix_001/report 的四个汇总产物纳入 Git
  EXECUTION_001.md              真实 A800 提取、CPU 实验与结果解释
  verification/                准备/执行审计、单元测试日志与 Smoke 收据
```

输入真实字段已核实：

- Dataset：`question_id`、`question_text`、`type`。原 Query 文本保留，包括已有空格；按 ID 升序导出，只含两个允许字段。
- 正式曲线：`scores[str(k)].f1`、`context_tokens[str(k)]`、`api_token_usage[str(k)].prompt_tokens`。
- Evidence 来自原 `evidence_diagnostic/per_k_evidence.csv` 的 `sentence_recall_lower`，不重新匹配证据。

准备结果：1000 个唯一非空 Query，500 Bridge + 500 Comparison，ID 集与旧 Soft Oracle 完全一致；752 个有效标签（296 Bridge、456 Comparison），181 个 `all_zero_f1`，67 个 `empty_candidates`。三种目标共用有效状态，248 个无标签目标为 `null`。

Dataset SHA256：`b9798320af36fbeedf1e81ed7690673016ba77b3cdc085d268c36b50669b04af`。

Query JSONL SHA256：`876d1ea0f5ad699a793a84ced0be45bd0a80c8277241140b2a64f86217d91304`。

| 划分（每个 seed） | 全部 | 有效标签 | 无标签 | Bridge / Comparison |
| --- | --- | --- | --- | --- |
| Train | 700 | 526 | 174 | 350 / 350 |
| Validation | 100 | 76 | 24 | 50 / 50 |
| Test | 200 | 150 | 50 | 100 / 100 |

种子为 13、42、2026。按 `type × has_valid_label` 四层确定 ID 归属，最大余数分配保证精确 70/10/20；之后训练按 ID 连接特征与目标。每个种子内互斥且覆盖全部 ID；各目标与架构共用对应划分。不同种子之间允许测试 ID 重叠。没有使用测试预算标签值进行分层或选参。

## 手动步骤 1：4090 上确认上传目录

以下命令中的仓库路径为当前实际路径；默认从此目录执行。

```sh
cd /media/hnu/hnu2021/dengxin/luoyichen/kbqa
ls -lh analysis/query_budget_pilot/staging/a800_embedding_job
```

已生成的准确相对路径：`analysis/query_budget_pilot/staging/a800_embedding_job/`，全部文件为：

```text
extract_embeddings.py
queries.jsonl
requirements-embedding.txt
README.md
a800_env.sh
job_manifest.json
```

`job_manifest.json` 保存其他 5 个文件的 SHA256、原始数据 SHA256、1000×1024 的预期规模。不包含答案、标签、历史 Checkpoint、索引、模型或 Embedding。可直接上传整个小目录。

数据已生成，无需重新执行。需要复现时，指定尚不存在的准备目录，并用新的配置文件把 `prepared` 改为该路径；依次运行：

```sh
python3 analysis/query_budget_pilot/prepare_queries.py --config /absolute/path/to/new_config.json
python3 analysis/query_budget_pilot/prepare_labels.py --config /absolute/path/to/new_config.json
python3 analysis/query_budget_pilot/pack_a800_job.py --config /absolute/path/to/new_config.json \
  --output analysis/query_budget_pilot/staging/a800_embedding_job_002
```

这些准备程序仅依赖标准库。既有输出会明确拒绝覆盖。

## 手动步骤 2：上传到 A800

用实际 SSH 信息替换 `<A800_USER>`、`<A800_HOST>`、`<A800_PORT>`，不要把尖括号原样保留。先在用户自己的 A800 终端执行：

```sh
mkdir -p /nas/luoyichen/query_budget_pilot
```

在 4090 终端执行（SCP 的端口参数是大写 `-P`）：

```sh
cd /media/hnu/hnu2021/dengxin/luoyichen/kbqa
scp -P <A800_PORT> -r analysis/query_budget_pilot/staging/a800_embedding_job \
  <A800_USER>@<A800_HOST>:/nas/luoyichen/query_budget_pilot/
```

若两台服务器不能互相 SSH，改在用户本地 **Windows PowerShell** 中转；另外替换 4090 的用户名、主机与端口。A800 上仍先创建上述目录：

```powershell
New-Item -ItemType Directory -Force "$HOME\query_budget_transfer" | Out-Null
scp -P <4090_PORT> -r <4090_USER>@<4090_HOST>:/media/hnu/hnu2021/dengxin/luoyichen/kbqa/analysis/query_budget_pilot/staging/a800_embedding_job "$HOME\query_budget_transfer\"
scp -P <A800_PORT> -r "$HOME\query_budget_transfer\a800_embedding_job" <A800_USER>@<A800_HOST>:/nas/luoyichen/query_budget_pilot/
```

## 手动步骤 3：A800 环境、提取与验证

所有下列命令在 A800 上手动运行；不需要克隆仓库。`a800_env.sh` 保证 Hugging Face、Sentence Transformers、PyTorch、pip、Python 字节码、CUDA 缓存和临时文件位于 `/nas/luoyichen/`。不会修改系统 HOME。

建议 Python 3.10/3.11。已有 `/nas/luoyichen/` 下可用的相同依赖环境时，可直接激活并跳过创建/安装；确认包版本和 `pip check`。

```sh
cd /nas/luoyichen/query_budget_pilot/a800_embedding_job
. ./a800_env.sh
python3 --version
mkdir -p /nas/luoyichen/envs
python3 -m venv /nas/luoyichen/envs/query-budget-embedding
. /nas/luoyichen/envs/query-budget-embedding/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-embedding.txt
python -m pip check
python -c 'import torch, numpy, sentence_transformers; print(torch.__version__, numpy.__version__, sentence_transformers.__version__)'
nvidia-smi
```

示例使用 CUDA 12.1 的 PyTorch wheel；按服务器驱动和已有环境选择可用版本。A800 上的安装由用户执行，当前 4090 没有安装提取依赖或下载 BGE-M3。上传包 README 另含逐文件哈希和 Query 检查命令。

用户确认空闲物理 GPU 后，替换 `<GPU_INDEX>`。只暴露一张卡，进程中的 `cuda:0` 对应所选物理卡：

```sh
GPU_INDEX='<GPU_INDEX>'
mkdir -p /nas/luoyichen/query_budget_pilot/results/run_001
CUDA_VISIBLE_DEVICES="$GPU_INDEX" python extract_embeddings.py \
  --input /nas/luoyichen/query_budget_pilot/a800_embedding_job/queries.jsonl \
  --output /nas/luoyichen/query_budget_pilot/results/run_001/embeddings.npz \
  --model BAAI/bge-m3 --batch-size 32 --device cuda:0 \
  --cache-folder /nas/luoyichen/models/sentence-transformers
```

该方式复用指定缓存，缺少文件时才会下载。已有本地 BGE-M3 的替代命令（与上一条二选一，替换实际模型目录）：

```sh
CUDA_VISIBLE_DEVICES="$GPU_INDEX" HF_HUB_OFFLINE=1 python extract_embeddings.py \
  --input /nas/luoyichen/query_budget_pilot/a800_embedding_job/queries.jsonl \
  --output /nas/luoyichen/query_budget_pilot/results/run_001/embeddings.npz \
  --model /nas/luoyichen/models/bge-m3 --local-files-only \
  --batch-size 32 --device cuda:0
```

本地目录须包含官方 Sentence Transformers CLS pooling 配置。可用 `--revision <MODEL_COMMIT>` 固定 Hub 模型版本；元数据记录可获取的实际 commit、本地配置 SHA256、软件版本、模型配置、脚本 SHA256 与命令。完整 API 参考：[官方 BGE-M3 模型说明](https://huggingface.co/BAAI/bge-m3)、[所固定的 Sentence Transformers 3.4.1 源码](https://github.com/UKPLab/sentence-transformers/blob/v3.4.1/sentence_transformers/SentenceTransformer.py)。

默认冻结 BGE-M3、`eval()`、`inference_mode()`、float32 前向，仅 Dense、CLS pooling、L2 归一化，不添加检索指令；按 batch 编码。最大长度 8192，检测到 Query 会被截断时直接报错。`--no-normalize` 可用于单独提取实验，但本 Pilot 训练明确要求 L2 归一化。

提取后检查维度、ID 数量、有限值、归一化、输入/输出哈希和全部元数据，**此命令不加载模型或使用 GPU**：

```sh
python extract_embeddings.py \
  --input /nas/luoyichen/query_budget_pilot/a800_embedding_job/queries.jsonl \
  --output /nas/luoyichen/query_budget_pilot/results/run_001/embeddings.npz \
  --verify-only
```

预期 `shape=[1000,1024]`、`question_id_count=1000`。NPZ 中严格含 Unicode `question_ids` 和 float32 `embeddings`，无需 pickle。每行向量与输入 ID 一同保存；训练重新按 ID 对齐，绝不假设 NPZ 与标签的行顺序相同。

## 手动步骤 4：传回 4090

在 4090 终端执行：

```sh
cd /media/hnu/hnu2021/dengxin/luoyichen/kbqa
mkdir -p analysis/query_budget_pilot/inputs
scp -P <A800_PORT> \
  <A800_USER>@<A800_HOST>:/nas/luoyichen/query_budget_pilot/results/run_001/embeddings.npz \
  analysis/query_budget_pilot/inputs/
scp -P <A800_PORT> \
  <A800_USER>@<A800_HOST>:/nas/luoyichen/query_budget_pilot/results/run_001/embedding_metadata.json \
  analysis/query_budget_pilot/inputs/
```

不能直连时，改在 Windows PowerShell 下载并上传这两个文件：

```powershell
New-Item -ItemType Directory -Force "$HOME\query_budget_transfer\results" | Out-Null
scp -P <A800_PORT> <A800_USER>@<A800_HOST>:/nas/luoyichen/query_budget_pilot/results/run_001/embeddings.npz "$HOME\query_budget_transfer\results\"
scp -P <A800_PORT> <A800_USER>@<A800_HOST>:/nas/luoyichen/query_budget_pilot/results/run_001/embedding_metadata.json "$HOME\query_budget_transfer\results\"
scp -P <4090_PORT> "$HOME\query_budget_transfer\results\embeddings.npz" "$HOME\query_budget_transfer\results\embedding_metadata.json" <4090_USER>@<4090_HOST>:/media/hnu/hnu2021/dengxin/luoyichen/kbqa/analysis/query_budget_pilot/inputs/
```

`inputs/` 已在仓库中准备。回传后先在 4090 校验（只需 numpy）：

```sh
cd /media/hnu/hnu2021/dengxin/luoyichen/kbqa
/home/dengxin/miniconda3/envs/adaptiverag/bin/python analysis/query_budget_pilot/extract_embeddings.py \
  --input analysis/query_budget_pilot/data/prepared_001/queries.jsonl \
  --output analysis/query_budget_pilot/inputs/embeddings.npz --verify-only
```

## 手动步骤 5：4090 上仅 CPU 训练与离线评价

已有 `adaptiverag` Python 3.8.20 / torch 1.13.1 / numpy 1.24.4 / matplotlib 3.7.5 可运行训练，测试已在这个环境完成。新建环境时安装 `requirements-training.txt`；仅 CPU wheel 足够。不要在当前环境安装 A800 的提取依赖。

一条命令运行全部 3 标签 × 2 架构 × 3 seeds，并自动评价与生成报告：

```sh
cd /media/hnu/hnu2021/dengxin/luoyichen/kbqa
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/dengxin/miniconda3/envs/adaptiverag/bin/python analysis/query_budget_pilot/run_matrix.py \
  --config analysis/query_budget_pilot/config.json --device cpu \
  --output analysis/query_budget_pilot/outputs/matrix_001
```

CPU 线程数默认 2；可用 `--num-threads 1` 降低 CPU 占用。代码只接受 `--device cpu`，不会把模型或张量放到 4090。缺少真实 NPZ 或元数据会立即明确提示先手动执行 A800 任务，不下载模型，也不生成正式训练目录。

单次训练与评价示例：

```sh
CUDA_VISIBLE_DEVICES='' /home/dengxin/miniconda3/envs/adaptiverag/bin/python analysis/query_budget_pilot/train.py \
  --model linear --label minimum_feasible --seed 13 --device cpu \
  --output analysis/query_budget_pilot/outputs/single_001
CUDA_VISIBLE_DEVICES='' /home/dengxin/miniconda3/envs/adaptiverag/bin/python analysis/query_budget_pilot/evaluate.py \
  --run analysis/query_budget_pilot/outputs/single_001
```

`--model` 为 `linear` 或 `mlp`；`--label` 为 `minimum_feasible`、`original_soft` 或 `sharper_soft`；seed 选已保存的 13、42、2026。训练支持 `--epochs`、`--batch-size`、`--learning-rate`、`--weight-decay`、`--patience`、`--num-threads`，默认见 config。也支持 `--prepared`、`--embeddings`、`--embedding-metadata` 自定义路径，三者要匹配。

批量入口已自动生成 `outputs/matrix_001/report/report.md`、`summary.json`、`summary.csv`、`f1_token.png`。需要在新目录重新汇总已有评价时：

```sh
CUDA_VISIBLE_DEVICES='' /home/dengxin/miniconda3/envs/adaptiverag/bin/python analysis/query_budget_pilot/evaluate.py \
  --matrix analysis/query_budget_pilot/outputs/matrix_001 \
  --output analysis/query_budget_pilot/outputs/matrix_001/report_002
```

所有训练、评价、打包目录存在即拒绝覆盖。省略训练/批量入口的 `--output` 会分配新的 `train_NNN` / `matrix_NNN`。运行中失败的目录保留；用新目录重跑。Embedding、权重、逐次训练中间产物和运行缓存被本目录 `.gitignore` 忽略；`outputs/matrix_001/report/` 的报告、JSON/CSV 汇总及 PNG 对比图四个产物纳入 Git。

## 方法、评价与解释边界

标签构造直接导入原 `build_soft_oracle.build_label` 和 `followup_diagnostics.analyze.min_feasible_k`。ε=0.1、τ=0.5、方案 D、Evidence=`sentence_recall_lower`、Cost=`context_tokens`、k=1…15。成本按每题全部 15 点的最大值归一化，不重新实现候选筛选或 Softmax。准备时逐题核对旧 1000 个 Soft Label 和 Minimum Feasible 结果，包括 248 个无效状态。

| 训练目标 | 参数与定义 |
| --- | --- |
| minimum_feasible | 真实候选中 Context Tokens 最少，成本并列取较小 k，One-hot；不假设成本单调 |
| original_soft | λ=0.1、T=0.1，原双约束 Soft Label |
| sharper_soft | λ=0.2、T=0.02，相同候选与有效状态 |

输入只为 1024 维 Query Embedding。题型、有效状态仅用于划分或诊断，F1、Evidence、k、Token、答案不进入特征。Linear 为 `Linear(1024,15)`；MLP 为 `Linear(1024,256) → ReLU → Dropout(0.2) → Linear(256,15)`。默认 AdamW、batch 64、weight decay 0.01、Linear LR 0.01 / MLP LR 0.001、最多 200 轮、patience 20、min_delta 1e-4，保存最佳有效 Validation Soft CE 检查点。

Checkpoint 保存张量，架构/设置/来源记录在 `training.json`；评价以 `weights_only=True` 和 `map_location='cpu'` 加载。训练过程不计算 Test loss；不以测试集选参数或固定 k。不同监督目标的 CE 含不同目标熵，不能直接按跨目标 Validation Loss 排名。

每次评价输出全部 1000 ID 的概率、预测 k、归属划分和标签状态。主指标在完整的 Train/Validation/Test 上分别查表：Answer F1、Context Tokens、Prompt Tokens、Evidence Recall、各 k 分布、k1比例；包含 Bridge/Comparison 及有效/无标签辅助分组。

固定基线为预先指定的 k=1/3/5/13；不搜索测试最优 k。k=13 在原整批 Dev 分析中曾表现最好，本 Pilot 对它的比较仍属于探索性。`offline_max_f1_oracle` 读取真实逐 k 分数，是保存结果表上的 F1 参考上界，不能在线部署，也不代表成本最优。

汇总先在每个 seed 相同测试集上计算相对 Fixed k 的差，再报告跨三个种子的均值与**样本标准差**。种子同时改变划分和初始化，波动不是独立置信区间。预测 ≥95% 同一 k 的 seed 记描述性塌缩标志；结合分布与 F1–Token 图判断，不以 Oracle k 分类准确率作结论。现有 1000 题已用于方法设计，不能宣称独立泛化收益。

## 测试与交付核验

```sh
cd /media/hnu/hnu2021/dengxin/luoyichen/kbqa
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dengxin/miniconda3/envs/adaptiverag/bin/python -m unittest discover \
  -s analysis/query_budget_pilot/tests -p 'test_*.py' -v
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/dengxin/miniconda3/envs/adaptiverag/bin/python analysis/query_budget_pilot/smoke_test.py \
  --output analysis/query_budget_pilot/outputs/smoke_002
/home/dengxin/miniconda3/envs/adaptiverag/bin/python -m black --check analysis/query_budget_pilot
/home/dengxin/miniconda3/envs/adaptiverag/bin/python -m ruff check \
  --config analysis/query_budget_pilot/ruff.toml analysis/query_budget_pilot
git diff --check
```

单元测试涵盖数据污染字段、重复 ID、缺失/非有限/错误维度向量、哈希、重排行对齐、非单调最低成本、无效标签、划分互斥、可复现、Early Stopping、完整无标签测试覆盖及正式逐 k 查表。专门测试改变 Test 标签与 F1 后最佳训练权重完全一致；独立复制脚本的 `--verify-only` 无需仓库或 Sentence Transformers。提取路径用假编码器验证冻结、推理模式、API 参数与文件导出，**没有加载真实 BGE-M3**。

本次交付的 17 项单元测试全部通过；Smoke 完成 18 次合成数据 CPU 实验及报告生成。Black、Ruff、Shell 语法和 Python 编译检查通过。

Smoke 使用 100 个明确标为 SYNTHETIC 的问题、随机 L2 向量和合成结果表，跑完 18 次短训练、评价、聚合、绘图与报告。临时模型和向量随临时目录删除，只保留收据；没有把随机向量配给真实 HotpotQA 标签，没有正式实验指标。真实 BGE-M3 前向仍需用户在 A800 执行后验证。

`verification/` 保存检查日志、Smoke 收据、文件清单及旧文件 SHA256 审计。原有 RAG、Soft Oracle、后续诊断结果及用户已有未跟踪文档保持不变。准备阶段按要求停在上传前；获得后续 SSH 授权后已完成真实提取和全部预设 CPU 实验，详见执行记录。
