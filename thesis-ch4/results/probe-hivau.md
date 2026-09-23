# HIVAU-70k 时序标注前置探查

## 结论

存在事件级时序起止标注，SPEC-01 §3.2A 的停止条件未触发。

- 标注目录：`/root/autodl-tmp/HolmesVAU-master/HIVAU-70k/raw_annotations/`
- 事件级边界字段：`events`，结构为 `[[start_sec, end_sec], ...]`。
- 时间单位：`events` 使用秒。UCF-Crime 与 XD-Violence 均有逐视频 `fps` 字段；XD-Violence 另有冗余的 `events(frame)` 帧边界。
- 帧率来源：每条视频记录自身的 `fps`；不应假设全库固定帧率。
- `clips` 是事件内部更细片段，不能替代事件级 `events`。
- `instruction/*.jsonl` 的 `type: "event"` 是已裁切事件视频的指令数据，不是原始长视频边界的事实来源。

## 统计

遍历四个原始 JSON 的全部记录；每个 `events` 区间计一个事件，时长为 `end-start` 秒。

| 标注文件 | 记录数 | 含事件边界视频数 | 事件数 | 含 `events(frame)` 记录数 | 时长 min / median / max（秒） |
|---|---:|---:|---:|---:|---:|
| `ucf_database_train.json` | 1,493 | 1,493 | 2,840 | 0 | 1.000 / 15.0495 / 1302.433 |
| `ucf_database_test.json` | 251 | 251 | 337 | 0 | 1.000 / 14.034 / 1722.634 |
| `xd_database_train.json` | 3,950 | 3,950 | 8,236 | 3,950 | 1.000 / 27.625 / 4280.667 |
| `xd_database_test.json` | 800 | 800 | 1,838 | 800 | 0.292 / 11.125 / 625.000 |
| **合计** | **6,494** | **6,494** | **13,251** | **4,750** | **0.292 / 21.625 / 4280.667** |

## 3 条真实原始 JSON 片段

以下只裁出与边界解释直接相关的原字段，数值未改写。

### 样例 1

来源：`raw_annotations/ucf_database_train.json`

```json
{
  "Abuse001_x264": {
    "n_frames": 2729,
    "fps": 30.0,
    "label": ["Abuse"],
    "events": [[3.933, 15.867], [24.7, 42.5]]
  }
}
```

### 样例 2

来源：`raw_annotations/ucf_database_test.json`

```json
{
  "Arrest001_x264": {
    "n_frames": 2374,
    "fps": 30.0,
    "label": ["Arrest"],
    "events": [[39.5, 49.5]]
  }
}
```

### 样例 3

来源：`raw_annotations/xd_database_test.json`

```json
{
  "v=S-7rRLrxnVQ__#1_label_B4-0-0": {
    "n_frames": 3273,
    "fps": 24.0,
    "label": ["Riot"],
    "events": [[0.0, 63.208], [82.083, 126.583]],
    "events(frame)": [[0, 1517], [1970, 3038]]
  }
}
```

## 视频资产可用性附注

`HIVAU-70k/videos/ucf-crime/videos` 与 `HIVAU-70k/videos/xd-violence/videos` 是普通文本占位文件，内容指向旧机器的 `/home/dancer/MLLM/HolmesVAU_full/HIVAU/videos/...`，不是目录或软链接。当前数据盘未找到对应原视频。时序边界本身存在，所以 §3.2A 停止条件未触发；但原视频缺失阻塞 §3.4，详见 `issues/01-hivau-videos-missing.md`。
