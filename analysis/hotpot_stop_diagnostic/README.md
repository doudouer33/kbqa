# Qwen HotpotQA stop 小样本诊断

本诊断使用仓库 `run.py write / predict`，再由 `predict.py` 调用原有
`commaqa.inference.configurable_inference`。新增的专用配置保留历史
`predictions/dev_1000` 的实验设置，唯一生成参数修改为 `stop=[]`。
其他模型和原有配置不变。固定读取原文件中的前 20 道 bridge 和前 20 道 comparison，
只运行 k=1、8、15。

现有 `run.py verify` 在第 853 行引用未定义的 `args`，准备阶段即报错；本诊断
保留该代码，以 Jsonnet 实际求值与历史配置相等性检查替代这一步，不影响生成。

```sh
/home/dengxin/miniconda3/envs/adaptiverag/bin/python \
  analysis/hotpot_stop_diagnostic/run_diagnostic.py \
  --set-name hotpot_stop_40_UNIQUE_RUN_NAME

/home/dengxin/miniconda3/envs/adaptiverag/bin/python \
  analysis/hotpot_stop_diagnostic/analyze.py \
  predictions/hotpot_stop_40_UNIQUE_RUN_NAME
```

`set-name` 对应的输出目录必须不存在；已有结果不会覆盖。具体 `run.py` 命令、
模型列表、40 个 IDs、历史文件校验值和环境适配保存在 `manifest.json`。

`sitecustomize.py` 仅在显式设置 `KBQA_STOP_DIAGNOSTIC_DIR` 且子进程包含
`--config --input` 时加载。钩子调用原函数并原样返回，仅记录 HTTP 请求/响应、
prompt 裁剪前后、原始 CoT、序列化后的提取器输入和最终答案。
生成调用通过缓存函数的 `__wrapped__` 执行原 API 调用体，绕过历史生成缓存；
每道题均须有一次真实请求。tokenizer 文件缓存仍可使用。

离线评分同时保存仓库 DROP 风格 F1 与 HotpotQA 官方 F1。`primary_match` 表示
现有提取器的首选正则命中，不等价于答案正确或提取结果干净；`marker` 仅供离线
观察，不会参与生成、提取或修改评分输入。旧输出从历史 chains 文件还原，
旧 finish_reason / token 数没有保存，不能从旧文本反推。
