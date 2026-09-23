# SPEC-04 · 数据地基 + B0（自采）

```
状态: draft
冻结日期: 2026-09-xx
依赖: SPEC-01（frozen：环境、HolmesVAU-2B 推理、第三章资产、词表）
执行方: Codex (AutoDL / thesis-ch4)
```

## 1. 目标（一句话）

在自采高架监控数据上建立**可复现的数据地基**（统一时间轴 + manifest + 相机分组 + 防泄漏划分），并跑 **B0（零样本 HolmesVAU-2B 直答时间戳）作为行为探针与地板**——为后续标注（SPEC-05）与基线（SPEC-06）铺路。

## 2. 不做什么

- 不训练、不 LoRA、不做标注工具、不做车道图、不做证据场、不做候选生成、不做任何方法（后续 SPEC）。
- **不做完整 IoU 评测**：自采的第四章精细定位标注为空，B0 只作**行为探针 + 30 段 pilot 抽查**，正式 IoU 评测等 SPEC-05 标注建成。
- **不试图提升 B0**：解析失败率高、退化严重都是预期结论；不得改 prompt 去"调好"、不得加重试、不得后处理。
- 不修改 SPEC-01 已冻结的任何文件与接口；不修改 `/root/autodl-tmp` 下既有原始数据。

## 3. 前置探查（不满足则记 issue 并继续，不静默改动）

产出 `results/probe-selfdata.md`：

**A. 数据画像复核**
- 文件数、六类分布、时长分布（确认 >120s 极少）、是否存在画面机位叠字。
- 附与《数据集画像_2026-09-19》的差异（若有）。参考基线：约 6,217 段；二轮 1471 / 施工 1421 / 事故 180 / 停车 1481 / 抛洒 165 / 拥堵 1499。**数字对不上就记 issue，不改盘点结论。**

**B. 时间轴口径探查（B0 时间可信的前提）**
- 抽样 N=50，用 `ffprobe` 读每个流的 `start_time` / 首帧 PTS / `nb_frames` / `avg_frame_rate` / `time_base`。
- 统计：①流起点非零的比例；②container `duration` 与流解码时长不一致的比例。附 3 个真实样例的字段。
- 若两类问题比例显著（>10%），在报告里显式写出——**§5.1 的归一化因此是强制项，不是可选**。

## 4. 接口契约

**输入**：自采视频根目录（路径走配置，默认见 §5.0）。

**输出**：
```
data/manifest.jsonl      每行 {video_id, path, category, duration_sec, fps_used,
                                first_frame_pts, time_base, camera_group,
                                camera_group_conf, split, flags[]}
data/timeline_report.md  时间轴统一规则 + 受影响文件统计
data/camera_groups.json  {group_id: [video_id...]}，含 uncertain 列表
data/splits.json         {train:[...], val:[...], test:[...]}，按 camera_group + 源事件隔离
results/probe-selfdata.md
results/b0/metrics.json
results/b0/predictions.jsonl
results/b0/pilot30.json  （若做 §5.5）
results/accept-04.json
```
**CLI**：
```
python -m data.build_manifest --src X --out data/manifest.jsonl --config configs/data.yaml
python -m data.make_splits    --manifest data/manifest.jsonl --out data/splits.json --config configs/data.yaml
python -m eval.b0_selfdata    --manifest data/manifest.jsonl --split test --out results/b0 --config configs/b0_selfdata.yaml [--resume]
```

## 5. 算法规格（写死，不留发挥空间）

**5.0** 所有路径、阈值、比例、prompt 一律走 yaml，代码中不得出现硬编码常量。

**5.1 时间轴统一（核心）**
- 统一口径＝**首帧归零 + 秒**：对每个流，`t_norm(frame) = (pts(frame) − first_frame_pts) × time_base`，单位秒。
- `manifest.duration_sec` 取**解码到的最后一帧的 t_norm**；当它与 container `duration` 不一致时**以流为准**，并在 `flags` 记 `dur_mismatch`。
- 流起点非零的文件，`flags` 记 `nonzero_start`。
- **下游所有时间戳（B0 输出、SPEC-05 标注、评测）一律使用此口径**——这是全章时间可比的唯一基准。

**5.2 相机分组（无结构化相机 ID）**
- 从固定 ROI（画面角落叠字，ROI 坐标走配置）裁剪 → OCR（引擎走配置）→ 归一化字符串（去空格/统一全半角）→ 精确匹配聚为 `camera_group`；近似串用编辑距离阈值（走配置）合并。
- OCR 置信度低于阈值、或聚类边界样本：`camera_group = "UNKNOWN_<i>"`、`camera_group_conf` 低、并进 `uncertain` 列表供人工复核。**不得臆造相机 ID。**

**5.3 防泄漏划分**
- **划分单位是 `camera_group`，不是单个视频**：同一 group 的所有视频进同一 split。
- 若能从文件名/时间戳推断"同一源事件的相邻切片"，强制进同一 split（规则走配置；无法推断则仅按 group）。
- 比例走配置（默认 70/15/15）；`test` 一经生成写入并锁定（再次运行不得变动 test 成员，除非显式 `--reseed`）。
- 输出**六类 × 三 split 的数量表**；稀有类（事故/抛洒）若某 split 为 0，记 issue（提示需分层抽样，交人决策）。

**5.4 B0（自采，行为探针）**
- 零样本 HolmesVAU-2B，**直答、不开思维链**，贪心解码（`temperature=0`），复用 `src/model/holmesvau_infer.py`。
- prompt 写死在 `configs/b0_selfdata.yaml`；输入给模型**视频 + 原始类别 category 作为查询条件**（匹配最终"以 q 为条件"的任务设定；category 在此仅作条件，不作真值）。
- 输出要求"异常起止时间（秒）"；解析器抽 `(start, end)`；**解析失败记为失败样本，不重试、不丢弃**（解析失败率本身是结论）。
- 指标（**无 IoU**，因无真值）：`parse_fail_rate`、`format_ok_rate`、输出区间时长分布与起点分布、`avg_frames`（模型实际看的帧数）、`full_clip_ratio`（输出≈[0, T] 的比例，探测退化）。

**5.5 Pilot 抽查（建议做，30 段）**
- 从 `test` 随机抽 30 段（`seed` 走配置），由人**单人快速**标一个粗起止区间（仅 sanity check，**不是**正式标注，不进 SPEC-05 数据集）。
- 报这 30 段的粗 `mIoU` 与 `Δt_start` 中位数。
- **双重用途**：既给 B0 一个 IoU 数量级，又**验证 §5.1 时间轴统一的正确性**——若 Δt_start 普遍偏移一个近似固定量，说明时间轴口径仍有 bug，须回查 §5.1。

## 6. 验收标准（逐条写入 `results/accept-04.json`）

- [ ] `manifest.jsonl` 行数 == 自采视频数；每行字段完整无缺
- [ ] `timeline_report.md` 给出"流起点非零 / duration 不一致"的比例与 3 个真实样例
- [ ] 不变量：每个视频 first_frame 归零后最小时间戳 == 0；`duration_sec` > 0
- [ ] `camera_groups.json` 覆盖全部视频；`uncertain` 列表单列且可复核
- [ ] **无任一 camera_group 跨 split**：脚本自检输出 `cross_split_leak == 0`（硬门）
- [ ] 六类 × 三 split 数量表存在；事故/抛洒在各 split 的数量已报（为 0 则已记 issue）
- [ ] `results/b0/metrics.json` 含 `parse_fail_rate / format_ok_rate / 时长分布 / 起点分布 / avg_frames / full_clip_ratio`
- [ ] `results/b0/predictions.jsonl` 行数 == test 集视频数；每行 `{video_id, raw_output, parsed_start, parsed_end, parse_ok, frames_seen}`
- [ ] （若做）`results/b0/pilot30.json` 含粗 mIoU 与 Δt_start 中位数
- [ ] 在第三章目录执行 `git status --porcelain` 为空，结果贴进 accept-04.json

## 7. 禁止事项

- **不得安装 flash-attn**（沿用 SPEC-01：改 `sdpa` / `eager`）；不得升级 torch / transformers / peft / accelerate。
- **不得臆造 camera_id**；不确定一律 `UNKNOWN_<i>` + 进 uncertain。
- **不得用 container duration 覆盖流解码时长**（冲突以流为准 + 记 flag）。
- **不得让任何 camera_group 跨 split**（防泄漏是硬门）。
- **B0 不得为好看而改 prompt / 加重试 / 后处理**；不得试图提升 B0 指标。
- 不得修改 `/root/autodl-tmp` 下既有原始数据与 SPEC-01 已冻结文件。
- 超过 5 分钟的推理一律 `tmux` 后台 + 日志落盘。

## 8. 交付物

- 代码 + `tests/`（必须含：`cross_split_leak == 0` 单测、时间轴首帧归零单测、解析器失败样本记录单测）。
- `RUN.md`（复现步骤）。
- `results/probe-selfdata.md`、`data/timeline_report.md`、`results/accept-04.json`。
- commit 注明 `SPEC-04`；触发的任何 issue 在 commit 说明中列出。

---

## 附：本 SPEC 的边界与依赖（给作者/Codex 复查者，不进实现）

- **B0 为什么不算 IoU**：自采精定位标注为空。正式 IoU 评测在 SPEC-06（基线），依赖 SPEC-05（标注）。本 SPEC 的 B0 是行为探针 + 地板。
- **时间轴统一为什么是重头**：Codex 数据盘点指出部分文件流起点非零、媒体时间口径不一；不统一，秒级偏移会被误判为模型误差。§5.1 是全章时间可比的地基。
- **相机分组为什么只到叠字聚类 + uncertain**：无结构化相机 ID；分组是车道图（未来）与防泄漏划分的前提，但边界样本必须交人复核，不能臆造。
- **不在本 SPEC 的**：车道图、证据场、标注工具、任何训练。
