# SPEC-04 自采前置探查

输入：/root/autodl-tmp/高架桥数据/video_fixed
选择画像明确对应的 video_fixed 集合；不把 video/ 原片与派生副本相加。原始数据只读。

## A 数据画像复核
```json
{
  "file_count": 6217,
  "categories": {
    "二轮车辆闯入": 1471,
    "占道施工": 1421,
    "多车事故": 180,
    "异常停车": 1481,
    "抛洒物": 165,
    "拥堵": 1499
  },
  "baseline_matches": true,
  "container_duration_distribution_sec": {
    "n": 6215,
    "min": 4.92,
    "median": 59.96,
    "max": 1174.72
  },
  "over120_count": 3,
  "baseline_container_duration": {
    "min": 4.92,
    "median": 59.96,
    "max": 1174.72,
    "over120": 3
  },
  "header_errors": [
    {
      "video_id": "二轮车辆闯入_20230720073802_突发事件_6967422",
      "path": "/root/autodl-tmp/高架桥数据/video_fixed/二轮车辆闯入_20230720073802_突发事件_6967422.mp4",
      "error": "ffprobe decoding error: [h264 @ 0x563c12b8bec0] error while decoding MB 6 81, bytestream -28\n"
    },
    {
      "video_id": "占道施工_20230714234829_运行安全_1156096",
      "path": "/root/autodl-tmp/高架桥数据/video_fixed/占道施工_20230714234829_运行安全_1156096.mp4",
      "error": "ffprobe decoding error: [h264 @ 0x556329a92ec0] error while decoding MB 10 87, bytestream -18\n"
    }
  ]
}
```
与《数据集画像_2026-09-19》的文件数/六类分布比较如上；容器跨度统计仅用于画像对照，manifest 使用解码时间轴。
抽样 50 帧；高置信度机位 OCR 0；固定 ROI、原始 OCR 和置信度见 results/spec04/ocr.jsonl，裁剪帧见 results/spec04/probe-frames。
机位叠字尚未由高置信度 OCR 验证，已记 issue。

## B 时间轴探查
# SPEC-04 时间轴报告

首帧归零 + 秒：t_norm(frame) = (pts(frame) − first_frame_pts) × time_base。
duration_sec 严格取最后一帧 t_norm；不加一帧时长，不使用 container duration、nb_frames/fps 或 DTS 替代。
不一致按有理数精确比较；因此常见的一帧显示时长差也计入 dur_mismatch。

统计范围：§3 随机抽样；共 50，成功 0，错误 50。

```json
{
  "affected_counts": {
    "nonzero_start": 0,
    "dur_mismatch": 0
  },
  "ratios": {
    "nonzero_start": null,
    "dur_mismatch": null
  }
}
```

真实样例：
```json
[]
```
