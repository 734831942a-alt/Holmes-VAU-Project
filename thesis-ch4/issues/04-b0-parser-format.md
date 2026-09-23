# SPEC-04：B0 输出格式不匹配与解析口径待明确

状态：阻塞 B0 继续运行。用户要求先诊断，不再全量推理。

## 实际证据

已保存 312/932 条结果，现有解析器判定 312 条失败，format_ok 为 0。离线逐条审核发现：

| 原始输出形态 | 数量 |
| --- | ---: |
| 明确且合法的秒数起止区间（英文自然语言） | 188 |
| 没有明确起止区间 | 117 |
| 只给一个发生时间 | 5 |
| 起止相同的零时长区间，结束单位省略 | 1 |
| `21.30:00` 等口径不明的时间表达 | 1 |

188/312（60.3%）的区间满足 `0 <= start < end <= duration_sec`，但这不证明定位正确：没有精细真值，不计算 IoU。186 条是 starts/begins at … seconds and ends at … seconds，另两条分别是自然语言范围和 respectively 表达。例如：

> The anomaly event starts at 6.12 seconds and ends at 8.56 seconds.

正式 parse_regex 仅接受 `START=…, END=…`。因此 100% 是当前严格解析协议下的失败率，不能解释成模型完全不会输出时间。SPEC-01 的原解析器也采用 START/END 语法；本次没有修改其冻结文件。SPEC-04 §5.4 分别要求 parse_fail_rate 和 format_ok_rate，但没有明确规定自然语言时间是否属于可解析范围。不能在看到结果后私自扩大正式解析规则并覆盖原指标。

另外 27 条原文经同一 tokenizer 重编码恰好为配置上限 32 tokens，且均没有句末标点；多条明显止于半句话。其中 26 条没有明确区间，1 条已经含完整自然语言数值范围但无句末标点。存在预算截断迹象，但冻结生成接口未保存生成 token IDs / EOS / stop reason，所以不能仅凭重编码认定 27 条均由预算截断。没有增加 token 预算或重新推理。

当前已完成样本仅有二轮 221 条、施工 91 条，是运行顺序的前缀，不是六类随机样本，不能外推全 test 集失败率。模型 chat 源码将 question 放入对话，并将 generation_config 传入 generate；静态检查没有发现 prompt 被替换或忽略的路径。

## 原因与处理

已确认主要混杂因素是输出格式与解析协议不匹配；同时存在无区间回答、非法/歧义时间和疑似截断。模型为何不遵循格式，仅凭这些输出不能进一步确定，不能归因为权重损坏或特定训练原因。

执行者应在早期持续格式失败时停下审查 raw_output，而不应把“失败可能是预期结论”当成继续消耗全量推理的理由。这是执行与监控判断失误。

保留原 prompt、解析器、token 预算、predictions、attempts 和 metrics；不重试、不后处理覆盖正式结果。离线诊断的候选数值仅用于辨别原因，不是修订后的 B0 指标。进一步修改正式解析口径涉及冻结纪律，须先明确其是否接受自然语言区间，再针对现有文本验证；本次不自动恢复推理。

可复现证据：`configs/diagnose04.yaml`、`eval/diagnose04.py`、`results/spec04/b0-diagnosis.json`。JSON 含每条证据、输入 SHA-256、类别分布、长度分布、未完成尝试；确认未改 B0 输入且未调用模型生成。

```bash
python -m eval.diagnose04 --config configs/diagnose04.yaml
```
