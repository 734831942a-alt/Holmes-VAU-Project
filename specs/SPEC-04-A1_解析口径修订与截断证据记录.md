# SPEC-04-A1 · 解析口径修订与截断证据记录（修订条款）

```
状态: frozen
冻结日期: 2026-09-21
修订对象: SPEC-04 §5.4、§6（增补/细化）；SPEC-01 不受影响
依据: 《Claude审阅意见_SPEC-04_B0》 + 作者 2026-09-21 三项批准
执行方: Codex-CLI
```

## 0. 作者裁决记录（DECISIONS）

- **① 原则上批准 parse/format 分离**：采纳 Layer A/B 语法（本条 §A）。语法规则为具体实现方案，作者可在冻结前对**单条规则**微调；分离方向已定，不再回摆。
- **② 暂不调大 max_new_tokens**：保持 32；调预算是另一个需单独授权的实验，本次不做。
- **③ 批准增加截断记录，但须满足 §C 的证据要求**：仅未来 run 生效，存量 312 条不回填。

---

## A. 修订 SPEC-04 §5.4：parse 与 format 分离（离线复算，不重推理）

**适用范围**：仅 SPEC-04 的 `b0_selfdata` 解析器；**SPEC-01 的严格 START/END 解析器不动**。

**执行方式**：对已存的 **312 条** `raw_output` **离线复算**（reparse_v2），**不调模型、不重生成**。原 `results/b0/predictions.jsonl`、`results/b0/metrics.json` **字节不可变**；新结果写 `results/spec04/reparse_v2/`，配置新增 `configs/b0_reparse_v2.yaml`（不改 `b0_selfdata.yaml`）。结果文件头必须注明："**312/932 顺序前缀、类别仅二轮 221 + 施工 91、不可外推全 test**"。

**Layer A — 抽取（是否找到成对秒数）**
- `parse_ok=true`：输出中**恰好一对**以"秒"为单位的非负十进制数值，可无歧义读作 (start, end)。覆盖：`START=x, END=y`（原格式）、`starts/begins at x seconds … ends at y seconds`、`from x to y seconds`、`x–y s / x-y 秒`。
- `parse_ok=false`，并各记**原因码**：
  - `no_interval`：无成对秒数；
  - `single_point`：只有一个时间值；
  - `multi_interval`：多于一对（**不取第一、不挑**，判歧义拒绝）；
  - `missing_unit`：某值缺单位且**不猜**就无法确定是秒（如 "…ends at 31"）；
  - `ambiguous_clock`：如 `21.30:00` 口径不明。
- 禁止：从上下文猜时间、补缺失值、裁剪到 [0,T]、按内容对错筛选。

**Layer B — 合法性（仅对 Layer A 通过者）**
- `valid`：`0 ≤ start ≤ end`；越界 `end > duration_sec` **不判失败**（记 `out_of_range=true` 进分布，不改值）；
- `reverse_order`（end<start）：**不交换**，记 Layer B 不合法、单列；
- `zero_duration`（start==end）：Layer A 通过、时长=0，进时长分布，不拒绝。

**修订后指标口径**
- `parse_fail_rate = 1 − Layer A 通过率`；
- `format_ok_rate` = 严格匹配 `format_regex`（精确 START/END）的比例——**定义不变**；
- **新增**原因码直方图（上列各类计数）；
- 判定为**测量定义变更**，非后处理（不改任何模型输出）。

---

## B. 修订 §5.4：生成预算保持不变

- `max_new_tokens = 32` 保持。**不得**为改善指标或"疑似截断"而悄悄加预算重跑。调预算须作者另行显式授权，且单独版本化为一次实验。

---

## C. 增补 §5.4：截断证据记录（**仅未来 run 生效，明确证据要求**）

未来任何调用生成的 run（B0 重跑、后续 SPEC 的推理）**必须逐样本持久化以下 authoritative 停止信号**，写入 `predictions.jsonl` / `attempts.jsonl`：

| 字段 | 定义 | 来源 |
|---|---|---|
| `stop_reason` | ∈ {`eos`, `max_new_tokens`, `stop_string`, `other`} | 生成 API，按下方判定序 |
| `generated_token_count` | 新生成 token 数 = `len(output_ids) − len(input_ids)` | 生成输出，非重编码文本 |
| `max_new_tokens` | 该次调用生效预算 | 运行配置 |
| `last_token_is_eos` | bool | 生成输出末 token == `eos_token_id` |
| `generated_token_ids` | 新生成的 token id 序列（≤32，全存） | 生成输出 |

**`stop_reason` 判定序（写死）**：
1. 生成 token 序列中出现 `eos_token_id` → `eos`；
2. 否则 `generated_token_count == max_new_tokens` → `max_new_tokens`；
3. 否则 → `other`。

**判定纪律（核心证据要求）**：
- `truncated=true` **只能**由 `stop_reason == max_new_tokens` 判定。
- **禁止**用"重编码长度==32 / 无句末标点"作为**认定**截断的依据——那只是启发式，仅可记为 `truncation_suspected`，且必须同时保留 `stop_reason`。
- 目的：让截断可被生成信号**证实**，而非事后从文本**猜测**。

**存量 312 条的处理**：当时的冻结生成接口未记录上述信号，故 **一律 `stop_reason = unknown` + `truncation_suspected`（启发式：retokenized_len==32 且无句末标点）**；**不得回填、不得改写成"确定截断"**。此为诚实记录，不是缺陷补救。

---

## D. 账本与验收（重申，不改）

- `n_attempted` / `n_persisted` / `n_interrupted` 三数分记；第 313 次无输出 = `interrupted`，**非 parse fail、不重试**。
- 验收"B0 完整覆盖 test"**保持 FAIL**（312/932）；`complete=false` 维持。
- **恢复全量 B0 需作者显式指示，且应在本修订实现之后**，避免再次全量烧在错的测量口径上。

---

## E. 验收标准（增补，逐条写 `results/spec04/accept-04-a1.json`）

- [ ] `reparse_v2` 在 312 条上跑出 Layer A 通过数（诊断已知 ≈188），并产出原因码直方图；
- [ ] 原 `results/b0/metrics.json`、`predictions.jsonl` **SHA-256 与修订前一致**（哈希比对入 accept）；
- [ ] `results/spec04/reparse_v2/` 结果文件头含"312/932 前缀、二轮221+施工91、不可外推"字样；
- [ ] `format_ok_rate` 仍为 0（真实发现：prompt 格式 0 遵守）；
- [ ] 单测覆盖 Layer A/B 每条允许/拒绝规则各一例：START/END、NL 区间、single_point、multi_interval、missing_unit、ambiguous_clock、reverse_order、zero_duration、out_of_range；
- [ ] 单测覆盖 `stop_reason` 判定序三分支（eos / max_new_tokens / other）；
- [ ] 存量 312 条的 `stop_reason` 全部为 `unknown`（不得出现 `max_new_tokens` 等被回填值）。

---

## F. 禁止事项（针对本修订）

- 不重推理、不恢复全量、不重试第 313 次。
- 不覆盖或修改原 `results/b0/*`、`configs/b0_selfdata.yaml`、SPEC-01 任何文件。
- 不调 `max_new_tokens`。
- 不把 `truncation_suspected` 记成"确定"，不回填存量样本的 `stop_reason`。
- 不因看到重算结果而反向修改 Layer A/B 规则去"凑"某个数。
