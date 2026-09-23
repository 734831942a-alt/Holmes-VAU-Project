# 第四章同步说明

服务器工作目录 `/root/thesis-ch4` 的可审阅镜像位于本仓库 `thesis-ch4/`。

每次第四章的代码、配置、测试、运行说明、冻结 SPEC、issue 或 JSON/Markdown 验收证据发生相关变更后，应同步到该镜像并提交推送，再交由 Claude 审阅。

不纳入 Git：数据集、视频、模型权重/检查点、缓存、日志、锁文件，以及完整 JSONL 预测。冻结 A1 审计的例外是 `results/b0/` 的 312 条 `predictions.jsonl`、`attempts.jsonl`，以及 `data/manifest.jsonl`、`splits.json`：它们是可复现性测试所需的冻结元数据夹具，不含原始视频。同步本身不授权启动、恢复或重试推理。
