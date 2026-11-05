# HotpotQA Train 5000：固定样本与 k=1–15 RAG 数据生成

本目录只负责采样和普通 RAG 数据生成。复用仓库 `run.py write`、`predict.py`、
BM25、ONER Reader、CoT、答案提取及原有 `~/.cache/gpt3calls` DiskCache。
不执行评分、Oracle、Soft Label、Evidence Recall 分析或模型训练，也不拆分 Chunk
或划分 Train/Validation/Test。默认命令只运行 Smoke Test。

## 已冻结的数据

`data/train_5000.jsonl` 已从 `processed_data/hotpotqa/train.jsonl` 的 90447 道题中抽取。
先排除正式配置所用的全部 15 个 Few-shot ID，剩余 90432 道作为候选池，
按源文件顺序使用 `random.Random(42).sample(..., 5000)` 无放回采样。
没有根据 level 或 type 筛选、配额或重采样。

| 分布 | 数量 |
| --- | ---: |
| easy | 929 |
| medium | 3221 |
| hard | 850 |
| bridge | 4012 |
| comparison | 988 |

每条记录直接保留源 JSONL 的原始字节，包括所有字段和嵌套内容。
顶层原有字段为 `dataset`、`question_id`、`question_text`、`level`、`type`、
`answers_objects`、`contexts`。`contexts` 是原预处理数据，推理时仍由正式 Reader
配置 `add_paras=false` 决定不将它直接作为检索上下文。

`data/sample_manifest.json` 记录源/样本/基线配置/Prompt 的 SHA256、种子、数量、
完整且有序的 ID 列表、源及样本的 level/type/交叉分布、排除的 Few-shot ID。
样本 SHA256：`fb4dce5fdabc5800bcce4b29d77579bc42dce697f8aa8b034e464fb53402fcf2`。

## 目录和配置

```text
experiments/hotpotqa_train_5000/
├── .gitignore
├── README.md
├── run.sh
├── settings.json
├── scripts/                 # common、sample、pipeline、hooks、sitecustomize、smoke_acceptance
├── tests/                   # test_pipeline.py
├── configs/                 # 原基线副本及 15 份原有 k 配置
├── data/                    # train_5000.jsonl、sample_manifest.json
└── runs/
    ├── run_manifest.json
    ├── selected_prompt.txt
    ├── validation_report.json
    ├── directory_migration.json
    ├── directory_migration/before/ # 迁移前脚本、入口及文档快照
    ├── logs/               # prepare、unit_tests、shell_smoke_recheck 等日志
    ├── smoke/              # 原 Smoke 数据、日志、Smoke/缓存验证报告
    └── predictions/        # 原 Smoke 预测、上下文、attempt 及逐题记录
```

这次仅移动目录并修改路径引用。既有数据、配置、运行 Manifest、测试报告和预测结果
原样保留，没有重新采样、生成配置或执行 RAG。旧 Manifest/报告中的路径作为历史记录保留，
脚本通过 `runs/directory_migration.json` 映射到新位置，并检查路径修改后脚本的哈希。
原运行 Manifest 的 SHA256 不变，因此已完成结果的实验身份、预测与上下文绑定也不变。
迁移前快照保存在 `runs/directory_migration/before/`，不作为当前代码导入。

`.gitignore` 忽略 Python 缓存、日志、锁、临时文件、JSONL 数据、生成的 k 配置及运行结果。
小型目录迁移记录保留为可提交文件，原有文件只是在本地被忽略，并未删除。

| 文件 | 用途 |
| --- | --- |
| `settings.json` | 固定数据来源、seed、数量、set_name、8002 端口、k 列表 |
| `scripts/sample.py` | 一次性冻结样本；已有样本时只核验，拒绝覆盖或修复不完整样本 |
| `data/train_5000.jsonl` / `data/sample_manifest.json` | 固定的正式样本与审计信息 |
| `scripts/pipeline.py` / `run.sh` | 配置生成、逐 k 调用原推理、严格完成校验、进度检查 |
| `configs/` | 基线副本及由 `run.py write` 生成的 15 份独立配置 |
| `runs/run_manifest.json` | 数据、配置、Prompt、代码、服务地址、Python/依赖版本的冻结信息 |
| `runs/selected_prompt.txt` | 由原 `read_prompt()` 选出的 15 个 Few-shot 文本 |
| `scripts/hooks.py` / `scripts/sitecustomize.py` | 仅本实验子进程启用的旁路观察记录 |
| `runs/smoke/smoke.jsonl` | 从固定 5000 题选出的 6 道覆盖样本，每个 level×type 组合 1 道 |
| `tests/test_pipeline.py` | 严格完成判定、数据格式和 Usage 来源的失败场景测试 |
| `scripts/smoke_acceptance.py` | 仅对 Smoke 数据执行真实推理、中断、恢复和跳过验收 |
| `runs/` 中的报告及日志 | 验收证据、准备及运行日志 |

基线为 `base_configs/oner_qa_qwen_hotpotqa_formal.jsonnet`。准备阶段将其按原文复制为
独立名称，调用 `run.py write --instantiation_scheme oner_qa --prompt_set 1
--topk_values 1-15 --llm_port_num 8002 --no_diff`，再复制生成结果到本目录。
独立配置名不会覆盖历史实验的配置。

逐份求值比较后，15 份配置除了 `generate_titles.retrieval_count=k` 均与基线完全一致。
同时已对照 `analysis/hotpot_formal_1000_20261014_run/configs/` 的 15 份历史正式配置。
Qwen2.5-7B-Instruct、BM25、15-shot Prompt、CoT、正则答案提取、temperature=0、
top_p=1、max_tokens=512、stop=[]、本地长度预算 16384 及检索过滤规则都沿用原实现。
模型 API 由环境变量 `OPENAI_BASE_URL` 指向现有服务的 **8002** 端口；原客户端本身不替换。

## 准备与核验

以下命令均从仓库根目录执行。使用已有 `adaptiverag` 环境，也可通过 `PYTHON` 指定
具备同样依赖的解释器；准备后恢复必须使用冻结的解释器与依赖版本。

```sh
export PYTHON=/home/dengxin/miniconda3/envs/adaptiverag/bin/python

# 本次已经完成；再次执行只验证既有样本，不重新随机采样。
"$PYTHON" experiments/hotpotqa_train_5000/scripts/sample.py

# 严格检查数量、唯一 ID、嵌套字段、Few-shot 排除、原始字节、seed 和哈希。
sh experiments/hotpotqa_train_5000/run.sh check-data

# 本次已经完成；首次生成配置，后续仅检查冻结状态，不运行推理。
sh experiments/hotpotqa_train_5000/run.sh prepare

"$PYTHON" experiments/hotpotqa_train_5000/tests/test_pipeline.py
```

`run.sh` 和 `scripts/pipeline.py prepare` 均不会调用采样函数。若样本、manifest 或代码不匹配，
会非零退出，要求保留原实验并为变更建立新的实验身份，不能直接混用已有结果。

## Smoke、正式启动和恢复

先确认现有 BM25 服务可用，且 8002 的 `/v1/models` 提供 `Qwen2.5-7B-Instruct`。
脚本自动对 localhost 关闭代理，并使用本地 tokenizer 缓存。

```sh
# 6 道题 × k=1、8、15；已完整的 k 严格验证后直接跳过。
sh experiments/hotpotqa_train_5000/run.sh smoke

# 初次验收：含真实 SIGTERM 中断和手动重跑的自动化演练；仅涉及 Smoke 数据。
"$PYTHON" experiments/hotpotqa_train_5000/scripts/smoke_acceptance.py

# 只有用户明确授权正式运行后才执行这一命令。
# 当前交付没有启动正式 5000×15。
sh experiments/hotpotqa_train_5000/run.sh full --allow-full-run
```

正式运行和恢复使用同一条 `full --allow-full-run` 命令，按 k=1 到 15 串行执行。
也可在明确授权后单独执行 `"$PYTHON" experiments/hotpotqa_train_5000/scripts/pipeline.py run-k --mode full --k 7 --allow-full-run`。
无需传额外 `--resume`；脚本自动检查已有结果。没有显式 full 参数不会启动正式数据推理。

实际逐 k 调用 `predict.py --skip-evaluation 1 --force --silent`。
这里 `--force` 只用于绕过旧的宽松完成检查；是否运行由本目录的严格校验先决定。
从不调用 `evaluate.py`，不要求 metrics 文件，也不使用 `run.py` 的 90% 判断。

完成须同时满足：推理子进程退出码为 0；预测 JSON 的 ID 集合与全部输入完全一致；
每个预测都是非空字符串；每题都有对应的当前 attempt 上下文记录；qid、k、预测、
chain、配置/数据/代码身份及所有最终文件 SHA256 均一致。通过后才原子写入 `complete.json`。
不能仅根据预测文件存在或旧 marker 存在认定完成。

未完成的 k 会整 k 重跑原流程，尽可能命中原 DiskCache。既有最终文件先归档到新
attempt 的 `previous_outputs/`，逐题记录和历史日志保留。每道完成题的观察记录原子保存并
fsync。中断、非零退出、空预测、HTTP 失败、异常 finish reason 或上下文无法可靠对应时
不生成完成 marker，并停止 Shell 后续 k。SIGTERM/Ctrl-C 会终止推理进程组；锁阻止
同一实验并发写入。SIGKILL 可能留下 `running` 状态，它仍视为未完成；存活子进程持锁时
会拒绝另一运行，需等原子进程结束再重跑。

恢复不保证对失联或未返回响应的请求“恰好调用一次”。若中断发生在响应写入 DiskCache
之前，该题可能再次请求模型；未知响应不伪造 Usage。

## 进度与输出

```sh
sh experiments/hotpotqa_train_5000/run.sh status --mode smoke
sh experiments/hotpotqa_train_5000/run.sh check --mode smoke
sh experiments/hotpotqa_train_5000/run.sh status --mode full
sh experiments/hotpotqa_train_5000/run.sh check --mode full --k 7
```

`status` 显示每个 k 最新 attempt 已落盘题数及严格完成状态；只读且不启动推理。
`check` 使用同样校验，发现任何指定 k 未完成则非零退出。

预测结果实际保存在 `runs/predictions/`；仓库原 `predictions/<set_name>` 路径通过兼容符号链接访问相同目录，原 `predict.py` 不变：

```text
experiments/hotpotqa_train_5000/runs/predictions/hotpotqa_train_5000_seed42/       # 正式，尚未启动
experiments/hotpotqa_train_5000/runs/predictions/hotpotqa_train_5000_seed42_smoke/ # 已保留的 Smoke
  run_identity.json
  oner_qa_qwen_hotpotqa_train_5000____prompt_set_1___bm25_retrieval_count__K___distractor_count__1/
    prediction__hotpotqa_to_hotpotqa__train_5000.json   # Smoke 的 basename 为 smoke
    prediction__..._chains.txt
    prediction__..._time_taken.txt
    prediction__..._full_eval_path.txt
    prediction__..._variable_replacements.json
    config__hotpotqa_to_hotpotqa__train_5000.jsonnet
    contexts.jsonl
    complete.json
    observations/QID.json
    attempts/UTC_TIMESTAMP_ID/
      attempt.json
      predict.log
      records/QID.json
      previous_outputs/                              # 重跑时可能存在
```

在每个 `attempts/.../predict.log` 中查看原始推理输出，以及逐题
`HOTPOT5000 k=... completed=... qid=... cache_hit=...` 日志。
`attempt.json` 保留 running/failed/complete 状态及失败原因。
全局准备日志在 `runs/logs/prepare.log`，验收日志在 `runs/smoke/smoke_acceptance.log`。

预测 JSON 格式完全沿用原框架：`{"qid": "extracted answer", ...}`。
`contexts.jsonl` 每行对应一个 `(qid,k)`，按输入顺序汇总，包含：

| 字段 | 含义 |
| --- | --- |
| `qid`, `k`, `prediction` | 问题、请求的 k、原答案提取器输出 |
| `binding` | 配置、输入、run manifest SHA256 及独立 set_name |
| `raw_contexts` | 原检索流程选入 Reader 的有序 title / paragraph_text |
| `reader_contexts` | 原 Reader 格式化并按原规则裁剪后传给 QA 的字符串列表 |
| `effective_context_text` | 在最终 post-fit Prompt 中逐字校验过的实际检索上下文，段间以两个换行连接 |
| `effective_context_verified_in_prompt` | 上下文确实出现在送入原 API/cache 调用的最终 Prompt 中 |
| `context_sha256`, `prompt_sha256`, `call_sha256` | 上下文、最终完整 Prompt、原调用参数（含 Prompt）的 SHA256 |
| `input_question`, `context_suffix` | 实际问题与原 QA 附加信息 |
| `prompt_cropped`, `fewshot_count_after_fit` | 原长度适配是否裁剪及最终 Few-shot 数量 |
| `raw_generation`, `chain`, `chain_sha256` | 原始输出与原推理链，供与预测核对 |
| `parameters` | 原生成调用参数，不含 Prompt 正文 |
| `cache_hit` | 本次调用是否直接命中原 DiskCache |
| `api` | 真实 Usage、来源、response ID/model 和 finish reasons；未知值为 null |

检索结果会经过原框架的超长段过滤、近重复过滤等处理，所以实际上下文数量可能小于 k。
原 Reader 对每段取前 350 个按空格分隔的词并附加标题；后续计算 Context Tokens 应使用
`effective_context_text`，Evidence Recall 可同时查看实际标题、原段落和裁剪后的文本。
这里没有预先计算这两项指标。

原 chain 文件只有检索标题和生成结果，不能替代实际上下文。旁路记录直接观察原
`LLMQAParticipantModel.query`、`LLMQAModel.ask_question` 和 post-fit `openai_call`
的输入，不另行检索、不重组 Prompt、不替换客户端、不绕过或更换 memoize 函数。
如果原长度适配裁掉了当前测试问题的上下文，实验会在 LLM 调用前报错，避免保存错误对应关系；
Few-shot 的原有裁剪逻辑仍然沿用并记录。

Usage 来源分为 `api_response`（本次返回的真实值）、`saved_api_response`（同一
call SHA256 和生成结果以前保存的真实值）、`unavailable_legacy_diskcache`（旧缓存无
Usage）或 `unavailable_api_response`（服务未给出有效 Usage）。
`saved_api_response` 不是本次新增调用费用；任何来源的 Usage 都不能直接累加当作恢复
过程中新增的总计费量。未使用 tokenizer 估算值填充 API Usage。

## 验收证据

`runs/logs/unit_tests.log` 保存 12 项失败场景检查；`runs/smoke/smoke_acceptance_report.json` 保存真实
6×3 Smoke、中断恢复、DiskCache 命中、预测与上下文一致性及严格跳过结果。
`runs/validation_report.json` 汇总历史配置对照、回归测试及最终核验。
原采样与 RAG 实现阶段验收已通过：12 项独立测试、6 项原有答案提取回归测试，以及覆盖全部 6 种
level×type 组合的 18 条真实预测。18 条均保留完整 15-shot、未触发 Prompt 裁剪，
上下文与最终 Prompt 对应，均有真实响应 Usage。k=8 中断前完成的 2 题在重跑时命中
原 DiskCache，保留原预测和真实 Usage。三个 k 再次运行均严格跳过。
`runs/smoke/cache_validation_report.json` 还逐条验证了 18 个最终 Prompt 的精确重建、原缓存键、
缓存生成文本和未改变的 legacy 返回结构。`runs/logs/shell_smoke_recheck.log` 记录 Shell 入口的
完整复验。正式输出目录尚未创建，5000×15 没有启动。
Smoke 完成后再次运行只做严格核验与跳过；要重做带中断的首次演练，应建立独立的新
实验身份，不要删除或篡改当前冻结记录。

原推理文件和历史实验均未修改。无需通过此实验目录导入任何 Oracle、标签或训练模块。

本次目录整理只运行了 Python 静态编译、导入及 Shell 语法检查、原有 12 项单元测试，
以及 `run.sh check --mode smoke` 对已保存 18 条预测的只读校验。没有启动 Smoke 或正式
RAG 推理，没有重新生成任何原有数据、Manifest、配置或报告。迁移前后哈希记录在
`runs/directory_migration.json`，本次轻量检查日志在 `runs/directory_migration/`。
