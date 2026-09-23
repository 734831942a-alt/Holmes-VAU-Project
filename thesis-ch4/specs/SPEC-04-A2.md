# SPEC-04-A2 · 全 test B0 运行（清洁重跑）

```
状态: frozen（内容已定稿）
运行授权: 已授权 —— 作者 2026-09-21 授权运行一次（本次为唯一一次全量运行）
定稿日期: 2026-09-21
依赖: SPEC-04、SPEC-04-A1（已接受 @f22c24c）、SPEC-01
执行方: Codex-CLI
```

## 0. 性质与授权边界

- **授权范围**：**一次**全 test（全 932）B0 **清洁重跑**，用 A1 修订后的解析口径 + 启用证据记录 + 预算仍 **32**。
- **内容已冻结**：§3–§8 的公式、字段、阈值、验收项均已定稿，Codex 不得改动。
- **运行已授权（仅一次）**：作者 2026-09-21 授权执行**一次**全量 B0 运行。**仅此一次**；本 run 完成后、未获新授权前，不得重跑或追加全量推理。
- A1 接受记录：`f22c24c` 已被接受为 A1 实现版本；本次运行意在把 **B0 全 test 覆盖**从 incomplete 补到 complete。

## 1. 目标（一句话）

对全 932 test **一次清洁重跑**，全部带**权威停止/截断证据**；旧 312（`results/b0/*`）保留为**不可变历史、只读作比对**，不复用为输入；并产出一份**可复现性交叉核对**。

## 2. 为什么是重跑而非复用（取舍留档）

- **B0 很便宜**（零样本、12 帧、32 token、贪心、2B），多跑 312 条代价可忽略。
- 换来两个实打实的好处：① **全 932 一致的权威截断/停止证据**（不再混合 provenance）；② **可复现性核对**——把重跑在 312 重叠视频上的输出与不可变原件比对，得到“B0 是否可复现”这个对全章可信度重要的结论。
- 且**更简单**：无复用/拼接逻辑、无混合 provenance、**第 313 次中断不再需要特殊处理**（它只是 932 里的一条）。
- 旧 312 不受影响：`results/b0/*` 逐字节不可变，作历史；重跑写**新目录**。

## 3. 旧结果不可变（明确）

- **受保护集** = A1 的 28 项 + A1 交付物（`results/spec04/reparse_v2/*`、`accept-04-a1.json`、`tests-04-a1.json`、`eval/generation_evidence.py`、`eval/reparse_v2.py`、`configs/b0_reparse_v2.yaml`、`specs/SPEC-04-A1.md`）+ **原 B0 的 `results/b0/*`（312、attempts、run.json、metrics）**。全部**逐字节不可变**（重跑不改，只**读**作 §5 比对）。
- 运行**前后各一次** protected 哈希审计，`unchanged==true` 为硬门。
- 重跑输出写**新目录**：`results/b0_full_test/`（生成产物）+ `results/spec04/b0_full_test/`（度量）。**不得写入或覆盖 `results/b0/*`**。
- 输出目录与任一受保护输入重叠即报错。

## 4. 全量清洁重跑 + 可续跑（run 内不重复推理）

- 对**全 932** test **逐条新推理**，带权威证据（`stop_reason / generated_token_count / generated_token_ids / last_token_is_eos`）；生成配置与冻结的 `configs/b0_selfdata.yaml` **完全一致**（prompt / 预算 32 / 温度 0 / 帧 12），不改。
- **增量持久化 + 可续跑**：每条生成后立即落盘；若本 run 被中断，已落盘算“已完成”，resume 只补剩余——**本 run 内不重复推理**。
- **第 313 次中断**：本设计**无需特殊处理**——它只是 932 里的一条，清洁重跑一次；旧的中断记录留在不可变 `results/b0/attempts.jsonl` 作历史。
- **账本**：`n_test=932`、`n_generated`（本 run 新生成数）、`n_interrupted`（本 run 未完成数）；`n_generated + n_interrupted == 932`；`complete=true` 仅当 `n_generated==932`。

## 5. 可复现性交叉核对（保留 —— 诊断，非 pass/fail 门）

- 对 312 重叠视频，比对**本次重跑输出** vs **不可变原 312**：
  - 报 `reproduced_exact_rate`（`raw_output` 逐字节一致比例）+ 不一致条目的差异摘要（video_id、两侧文本）。
- **这是报告诊断，不是门**：temp=0 应当确定，但若 GPU 栈非 bitwise 确定，不一致是“**可复现性发现**”，如实报、不当失败。

## 6. warmup 与监控（区分实现错误 vs 模型表现）+ 阈值

### 6.1 warmup 计入正式结果，不重复推理
- warmup **不是**丢弃式预跑，而是**正式 run 的前 `warmup_n` 条**：生成 → 立即落盘 → 暂停 → 检查 → OK 则继续（warmup 这批**保留为正式结果**）。因增量持久化，即便中止，已落盘这批也进入下次 resume 的“已完成”集——**不重复推理**。

### 6.2 监控如何区分（看原始证据，不只看聚合率）
- **判为“实现错误” → 暂停诊断（多半中止+修，须另一次授权）**：
  - `generation_exception_rate > 0`（generate/证据捕获抛异常、prefix 不匹配、count > budget）；
  - `evidence_malformed_rate > 0`（`stop_reason` 不在枚举、权威 token 证据缺失）；
  - `empty_raw_output_rate ≥ 0.50`（模型/解码产出空——像管线 bug）；
  - `truncated_rate ≥ 0.95`（**近乎全部**截断——像预算/配置被设错，而非模型行为）。
- **判为“模型表现” → 如实记录、不暂停**（是**发现**，不是 bug）：
  - `parse_fail_rate` 高、`format_ok_rate≈0`、**部分**截断、`no_interval` 占比高——**均不作暂停触发**。
  - 理由：格式不遵守、给不出干净区间是已知模型行为（前缀已见 `format_ok=0`、`parse_fail=0.40`）；事故/抛洒类 parse_fail 可能更高，那是**内容**，不是管线错。
- **黄金判据**：能被一眼原始输出推翻的“失败”（raw 明明有区间却记失败）＝**实现错**；raw 本身空/乱码＝**模型问题**。故监控在暂停判断前**必须抽看 `raw_sample_on_pause` 条 raw_output**。

### 6.3 阈值（已确认照用；全部 YAML 可配、运行前仍可再调）

| 参数 | 值 | 含义 |
|---|---|---|
| `warmup_n` | 16 | 首检查点样本数（至少跨两类） |
| `checkpoint_every` | 50 | 之后每 N 条复检一次 |
| `generation_exception_rate` 暂停 | `> 0` | 任何生成/证据异常即停 |
| `evidence_malformed_rate` 暂停 | `> 0` | 证据不合规即停 |
| `empty_raw_output_rate` 暂停 | `≥ 0.50` | 半数空输出＝疑似管线 |
| `truncated_rate` 暂停 | `≥ 0.95` | 近全截断＝疑似预算错（32 下正常会有**部分**截断，不在此列） |
| `raw_sample_on_pause` | 8 | 暂停判断前必看的原始输出条数 |

- **明确不设** `parse_fail` / `format_ok` 的暂停阈值——它们是模型行为，不是暂停条件。
- 监控只**暂停 + 诊断**，**不引入重试 / 丢弃 / 按结果挑样本**。

## 7. 度量（复用 A1，不新造口径）

- 生成阶段只落 `raw_output` + 证据；解析与指标用**已接受的 A1 `reparse_v2` 度量**跑全 932。
- **截断/停止证据现全 932 权威**（无混合 provenance）。
- 报告：全量 + **按类**——`parse_fail_rate`、`format_ok_rate`、原因码直方图、时长/起点分布、`out_of_range`、`full_clip_ratio`、`truncated`（全权威）、`avg_frames`。

## 8. 验收（机器可判定，写 `results/spec04/b0_full_test/accept-04-a2.json`）

- [ ] 覆盖 == 932 才 `complete=true`；否则 FAIL，记覆盖数与 interrupted 列表；
- [ ] 受保护集（A1 的 28 项 + A1 交付物 + 原 B0 `results/b0/*`）before/after `unchanged==true`；
- [ ] 全 932 `stop_reason ∈ {eos,max_new_tokens,other}` 且由权威信号判定；无 `unknown`（除非该条本 run 未生成/中断）；
- [ ] 可复现性核对已产出 `reproduced_exact_rate` 与差异摘要；
- [ ] 账本 `n_test/n_generated/n_interrupted` 自洽且和为 932；
- [ ] 输出只在新目录，`results/b0/*` 与 A1 交付物字节不变。

## 9. 禁止事项

- 不改 prompt / 预算 / 温度 / 帧数；不重试；不后处理；不覆盖既有结果；不改冻结文件、A1 交付物与原 B0 输出。
- **本授权仅限一次全量运行**；完成后未获新授权前，不得重跑或追加全量推理。
- 出现 §6.2 的“实现错误”信号或任何条款歧义时，暂停、写 `issues/04a2-*.md`、保留已落盘凭据，不自行改规则、不“凑”数。
