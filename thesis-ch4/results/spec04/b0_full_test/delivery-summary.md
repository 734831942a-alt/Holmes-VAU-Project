# SPEC-04-A2 交付说明

唯一授权 run：`30cbdfa7-cc9c-45f9-97e7-72f060b15d94`。全 932 条逐条新推理已完成；本次授权已消耗，不再追加推理。

## 验收

- §8 六项全部通过；complete=true。
- n_test=932，n_generated=932，n_interrupted=0；932 次调用、932 条落盘、932 个唯一 ID。
- 69 个受保护文件运行前后 SHA-256 一致；原 B0、A1、SPEC-01 与冻结配置未改。
- 20 个检查点，每次审看 8 条原文，共 160 条；所有实现错误暂停门均未触发。
- 5 项 A2 单测通过，真实模型调用数为 0（单测使用 mock）。实际 run 退出码为 0。
- 触发的 SPEC-04-A2 issue：无。

## 全量结果

- Layer A/B 有效区间：355/932；parse_fail_rate=61.91%。
- format_ok_rate=0%；保持冻结 prompt 与解析规则，未后处理。
- EOS 829 条；预算截断 103 条（11.05%）；无 unknown 或畸形证据。
- 解析失败原因：no_interval 543、single_point 30、missing_unit 3、ambiguous_clock 1。
- out_of_range=2，原样标记，未裁剪；full_clip_ratio=0；avg_frames=12。
- 以上是解析和输出行为统计，不是定位准确率。

| 类别 | 样本 | 有效区间 | 解析失败率 | 权威截断数 |
|---|---:|---:|---:|---:|
| 二轮车辆闯入 | 221 | 128 | 42.08% | 25 |
| 占道施工 | 223 | 161 | 27.80% | 3 |
| 多车事故 | 34 | 5 | 85.29% | 3 |
| 异常停车 | 210 | 3 | 98.57% | 0 |
| 抛洒物 | 21 | 1 | 95.24% | 2 |
| 拥堵 | 223 | 57 | 74.44% | 70 |

## 复现诊断

旧 312 条全部比对，312/312 raw_output 逐字节一致，reproduced_exact_rate=1.0，差异列表为空。这是本次运行的诊断，不是验收率阈值。旧 312 未用作推理结果输入。

## 文件

- accept-04-a2.json：逐条验收。
- metrics.json / parsed.json：全量、按类指标和逐条解析。
- reproducibility.json：312 条复现诊断及差异列表。
- protected-before.json / protected-after.json：保护审计。
- checkpoints/ / reviews/：中途原文、证据与审看决定。
- tests.json / delivery-verification.json：单测与交付复核。
- run.log / run.exit / state.json：后台日志与结束状态。
- results/b0_full_test/ 下的 predictions.jsonl、attempts.jsonl、run.json、launch.json：生成原件、调用账本及唯一 run 元信息。

运行说明见 RUN.md。代码复用冻结 A1 的 generation_evidence.py 与 reparse_v2.py，生成参数始终来自冻结 b0_selfdata.yaml。
