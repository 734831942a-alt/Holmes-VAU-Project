# SPEC-02 · 时序标注第一批：抽样、标注工具与 B0 评测

```
状态: frozen
冻结日期: 2026-09-14
依赖: SPEC-01（已完成，除 §3.4 B0 因 HIVAU 视频缺失而阻塞）
执行方: Codex (AutoDL / thesis-ch4)
```

## 0. 背景变更（必读）

SPEC-01 的探查确认了两件事，本 SPEC 据此改道：

1. **HIVAU-70k 的源视频在本机不存在**（`videos/` 下是指向旧机器路径的占位文本文件）。因此 B0 不再使用 HIVAU，改用高架桥自采数据。HIVAU 的标注仍保留，后续可能用于预训练，不在本 SPEC 范围。
2. **`video/` 中 1,191 个 ">60s" 文件多为时间基/容器错误造成的元数据假象**，`video_fixed/` 中仅 107 个真正 >60s。后续凡涉及时长的判断，一律以 `video_fixed/` 为准。

`video_fixed/` 的片段是未经"裁到只剩异常"的源片段，时长中位数 58.48s，适合作为时序定位的评测材料，**但没有任何时序起止标注**。本 SPEC 的核心就是补上第一批真值。

## 1. 目标

产出 60 段带人工时序真值的评测集、一个可持续使用的最小标注工具，并在这批数据上跑出 B0 地板指标。

## 2. 不做什么

- 不做训练、不做 LoRA
- 不实现证据曲线、候选生成、车道图
- **不得用任何模型自动生成或预填时序标注**。这一批是真值基准，必须全部人工标定；模型预标会造成循环论证
- 不修改 `/root/autodl-tmp/高架桥数据` 下任何既有文件
- 不尝试提升 B0 的指标

## 3. 任务

### 3.1 分层抽样

数据源：`/root/autodl-tmp/高架桥数据/video_fixed/`

抽样规则（`seed=42`，结果必须可复现）：

- 六类**各 10 段**，合计 60 段
- 时长约束：`25s <= duration <= 70s`
- **夜间覆盖**：每类至少 3 段为夜间。夜间判据取文件名中 `YYYYMMDDhhmmss` 的小时字段 ∈ [20, 23] ∪ [0, 5]
- 排除 27 个损坏源文件对应的条目、排除 `video_preview_10_per_class/` 与 `e8_multicar_case_study/` 中的派生副本
- 若某类在约束下不足 10 段，**不要放宽约束自行补齐**，按实际数量抽取并在报告中说明

产出 `data/manifest_batch1.jsonl`，每行：

```json
{"video_id": "...", "path": "...", "category": "...", "duration": 58.48,
 "fps": 25.0, "width": 2560, "height": 1440, "timestamp": "20250312213045",
 "is_night": true, "batch": "batch1"}
```

同时产出 `data/batch1_sampling_report.md`：六类实际抽取数、夜间数、时长分布，以及被排除的条目数与原因。

### 3.2 最小标注工具

**形态**：服务器上运行的本地 web 服务（标注者在 Mac 上通过 SSH 端口转发访问）。

```
ssh -p <端口> -L 8080:localhost:8080 root@<主机>
# 浏览器打开 http://localhost:8080
```

**功能范围**（不要超出）：

- 逐条播放 manifest 中的视频，显示当前时间、总时长、类别
- 键盘或按钮标记异常起点 / 终点，可回放校正
- 三个必填字段之外，提供两个标志位：
  - `start_observable`：异常的起始时刻是否在本片段内可见（事故常只拍到事后状态）
  - `spans_whole_clip`：异常是否从第 0 秒持续到片段结束
- 可写 `notes` 自由文本
- **自动记录每段的标注耗时**（进入到提交之间的秒数）
- **支持中断续标**：已标条目跳过，进度落盘

产出 `data/annotations_batch1.jsonl`，每行：

```json
{"video_id": "...", "category": "...", "duration": 58.48,
 "t_start": 12.4, "t_end": 31.8,
 "start_observable": true, "spans_whole_clip": false,
 "notes": "", "annotation_seconds": 96, "annotator": "..."}
```

代码放 `src/annotate/`，启动命令写进 `RUN.md`。

> 标注本身由人完成，不在 Codex 的任务范围内。Codex 交付工具并验证其可运行（用 2 条样例走通存取流程即可），然后停止，等待标注完成后再执行 §3.3。

### 3.3 B0 评测（标注完成后执行）

零样本直接问 HolmesVAU-2B 时间戳，**直答配置，不开思维链**，贪心解码。

- 数据：`data/annotations_batch1.jsonl` 的全部条目
- 提示模板写在 `configs/b0.yaml`，不硬编码
- 解析器从生成文本抽取时间区间；**解析失败记录为失败样本，不得丢弃、不得重试、不得后处理修正**

产出 `results/b0/metrics.json`，除常规指标外**必须包含异常占比统计**：

```json
{
  "n_samples": 60,
  "mIoU": ...,
  "R@0.3": ..., "R@0.5": ..., "R@0.7": ...,
  "parse_fail_rate": ...,
  "avg_frames": ...,
  "anomaly_ratio": {
    "definition": "(t_end - t_start) / duration",
    "min": ..., "p25": ..., "median": ..., "p75": ..., "max": ...,
    "spans_whole_clip_count": ...
  },
  "start_observable_false_count": ...,
  "by_category": { "<类别>": {"n": ..., "mIoU": ...} }
}
```

以及 `results/b0/predictions.jsonl`：
`{video_id, category, gt_start, gt_end, raw_output, parsed_start, parsed_end, iou, parse_ok}`

## 4. 配置项

`configs/sampling_batch1.yaml`（seed、每类数量、时长上下限、夜间小时区间、排除规则）
`configs/b0.yaml`（model_path、prompt_template、max_new_tokens、temperature=0、num_frames）

代码中不得出现硬编码常量。

## 5. 验收标准（写入 `results/accept-02.json`）

- [ ] `data/manifest_batch1.jsonl` 行数 == 60（若某类不足则等于实际总数，且报告中说明）
- [ ] 每类样本数 == 10（或说明不足原因）；每类 `is_night == true` 的数量 >= 3
- [ ] manifest 中所有 `duration` ∈ [25, 70]
- [ ] 重跑抽样脚本产出与首次**逐行一致**（seed 可复现）
- [ ] 标注工具能启动，且用 2 条样例完成"标注→落盘→重启后跳过已标条目"的往返
- [ ] `data/annotations_batch1.jsonl` 行数 == manifest 行数（§3.3 执行前的前提）
- [ ] `results/b0/metrics.json` 含上述全部字段，`anomaly_ratio` 与 `by_category` 齐全
- [ ] `results/b0/predictions.jsonl` 行数 == 标注条数
- [ ] `/root/autodl-tmp/高架桥数据` 下无任何文件被修改（执行前后 `find -newer` 对比，结果贴进 accept-02.json）

## 6. 禁止事项

- **不得用模型生成、预填或补全任何时序标注**
- 不得为了凑够每类 10 段而放宽时长或夜间约束
- 不得安装 flash-attn；不得升级 torch / transformers / peft / accelerate
- 不得修改、删除、移动 `/root/autodl-tmp` 下任何既有文件
- **不得试图提升 B0 的指标**。mIoU 低、解析失败率高都是预期结论，不得调 prompt、加重试、做后处理
- 超过 5 分钟的任务后台运行 + 日志落盘（本机无 tmux，可用 `nohup` 或 `setsid`，并在 RUN.md 中注明）

## 7. 交付

- 代码与结果 commit，信息注明 `SPEC-02`
- §3.1、§3.2 完成后即可交付并暂停，等待人工标注；§3.3 作为第二次提交
