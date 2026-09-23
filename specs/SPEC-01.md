# SPEC-01 · 环境快照、资产移植与 B0 基线

```
状态: frozen
冻结日期: 2026-09-14
依赖: 无
执行方: Codex (AutoDL / thesis-ch4)
```

## 1. 目标

把第三章的可复用资产以冻结引用的方式移植进本仓库，验证第四章所依赖的两项数据前提，并跑出 B0（零样本时序定位）这块地板指标。

## 2. 不做什么

- 不做任何训练、不做 LoRA 微调
- 不实现证据曲线、候选生成、车道图（后续 SPEC）
- 不修改 `/root/autodl-tmp/HolmesVAU-master` 下任何文件
- 不尝试提升 B0 的指标

## 3. 任务

### 3.1 环境快照

产出可重建环境的凭据（换实例时数据盘会清空，这是保险）：

```
results/env-snapshot/environment.yml     # conda env export --no-builds (env: holmesvau)
results/env-snapshot/requirements.txt    # pip freeze
results/env-snapshot/hardware.json       # nvidia-smi 关键字段 + torch/cuda 版本 + 显存
```

### 3.2 前置探查（不满足则停止）

**A. HIVAU-70k 的时序标注**

检查 `/root/autodl-tmp/HolmesVAU-master/HIVAU-70k/` 下 `raw_annotations/` 与 `instruction/`，回答：

- 是否存在**事件级（event-level）时序起止标注**？字段名是什么？
- 时间单位是秒还是帧？若是帧，fps 从哪里取？
- 统计：含事件级边界的视频数、事件总数、事件时长分布（min / 中位 / max）
- 附 3 条真实样例的原始 json 片段

产出 `results/probe-hivau.md`。

> **若不存在事件级时序边界：立即停止整个 SPEC**，写 `issues/01-hivau-no-temporal.md` 说明实际的标注结构，不要用片段级或视频级标注替代。

**B. 高架桥数据**

检查 `/root/autodl-tmp/高架桥数据` 与 `/root/autodl-tmp/HolmesVAU-master/autodl-tmp/高架桥数据`（确认两者是同一份还是不同），回答：

- 是否存在**未裁剪的原始长视频**（时长 > 60s）？数量、时长分布、分辨率、帧率
- 能否从文件名或目录结构推断相机数量 / 相机 ID
- 现有短片段的数量、时长分布、按六类的数量分布
- 是否存在夜间 / 雨天 / 遮挡的标记或可推断线索（文件名、目录、已有标注字段）

产出 `results/probe-bridge-data.md`。这一项即使结论为"无原始长视频"也继续执行 3.3 与 3.4，但要在报告里明确写出。

### 3.3 资产移植

从第三章目录 **copy**（不是 import、不是软链）到 `src/`，每个文件头加注释块，注明来源路径与 commit `8b98e6cda09d078061eb308d78feec75fa23e6fd`：

| 目标 | 来源 |
|---|---|
| `src/model/holmesvau_infer.py` | `inference.py` + `holmesvau/holmesvau_utils.py` 中的加载与生成部分 |
| `src/evidence/ovd_extract.py` | `ovd_video_summary.py` |
| `src/evidence/queries.py` | **三处词表合并**：`ovd_video_summary.py` 的 `TEXT_QUERIES` / `VSCM_CONTEXT_QUERIES` / `SARP_TARGET_QUERIES`，`eval_traffic_fixed.py` 的 `LABELS` / `ALIASES`，`build_ovd_prompt_zh.py` 的语义映射 |
| `src/eval/metrics_cls.py` | `eval_traffic_fixed.py` 中六类分类评测相关部分 |

要求：

- 移植后 `src/` **不得依赖第三章目录的任何路径**；模型权重、数据根目录一律走配置文件
- `src/evidence/queries.py` 是词表的**唯一事实来源**：六类标签、别名、每类对应的英文查询词、每类对应的语义提示，全部在此定义并导出。合并时若三处存在不一致，**不要自行裁决**——保留全部并在文件内注释标出差异，同时写进 `issues/01-vocab-conflict.md`

### 3.4 B0 基线

零样本直接问 HolmesVAU-2B 时间戳。**直答配置，不开思维链**（零样本 CoT 会显著拉低分数，把地板做虚低）。

- 数据：HIVAU-70k 中带事件级边界的视频，随机抽 **200** 段，`seed=42`
- 提示模板写死在 `configs/b0.yaml`，不在代码里硬编码
- 解码：贪心（temperature=0）
- 输出解析器：从生成文本中抽取时间区间。**解析失败必须记录为一条失败样本，不得丢弃或重试**——解析失败率本身是 B0 的结论之一
- 指标：`mIoU`、`R@1 IoU@{0.3, 0.5, 0.7}`、`parse_fail_rate`、`avg_frames`（模型实际看了多少帧）

产出：

```
results/b0/metrics.json       # 上述全部指标
results/b0/predictions.jsonl  # 每行 {video_id, gt_start, gt_end, raw_output, parsed_start, parsed_end, iou, parse_ok}
```

## 4. 配置项

`configs/b0.yaml`，全部键给出默认值与说明，代码中不得出现硬编码常量：

```yaml
model_path:      # HolmesVAU-2B 权重目录
data_root:       # HIVAU-70k 根目录
num_samples: 200
seed: 42
prompt_template: |
  ...
max_new_tokens:
temperature: 0
num_frames:      # 或 fps，取决于 inference.py 的实际接口
```

## 5. 验收标准（逐条写入 `results/accept-01.json`）

- [ ] `results/env-snapshot/` 三个文件存在且非空
- [ ] `results/probe-hivau.md` 明确给出字段名与时间单位，并附 3 条真实样例
- [ ] `results/probe-bridge-data.md` 给出原始长视频的存在性结论与时长分布
- [ ] `src/` 下四个模块存在，每个文件头含来源路径与 commit hash
- [ ] `python -c "from src.evidence.queries import LABELS; assert len(LABELS)==6"` 通过
- [ ] `grep -r "HolmesVAU-master" src/` 无结果（不残留第三章路径）
- [ ] `results/b0/metrics.json` 含全部六项指标
- [ ] `results/b0/predictions.jsonl` 行数 == 200
- [ ] 在第三章目录执行 `git status --porcelain` 输出为空，结果贴进 accept-01.json

## 6. 禁止事项

- **不得安装 flash-attn**。当前环境未装且可正常运行；若代码强制 require `flash_attn`，改为 `sdpa` 或 `eager` 注意力实现并在报告中记录，不要尝试编译安装
- 不得升级 torch / transformers / peft / accelerate
- 不得修改、删除、移动 `/root/autodl-tmp` 下任何既有文件（包括 `.Trash-0`、`hf_cache`、`高架桥数据`）
- **不得试图提升 B0 的指标**。B0 是地板，mIoU 低、解析失败率高都是预期结论；不得改 prompt 去"调好"、不得加重试、不得后处理修正
- 超过 5 分钟的推理一律 `tmux` 后台 + 日志落盘

## 7. 交付

- 代码与结果 commit，信息注明 `SPEC-01`
- 若触发了任何 issue，在 commit 说明里列出
