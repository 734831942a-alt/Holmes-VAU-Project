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
    "n": 6217,
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
  "header_errors": []
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
当前 FFmpeg 4.4 的解码帧 PTS 字段为 pkt_pts（配置 frame_pts_field）；不使用 pkt_dts 或 best_effort_timestamp 替代。

统计范围：§3 随机抽样；共 50，成功 50，错误 0。

```json
{
  "affected_counts": {
    "nonzero_start": 41,
    "dur_mismatch": 50
  },
  "ratios": {
    "nonzero_start": 0.82,
    "dur_mismatch": 1.0
  }
}
```

问题比例 >10%；§5.1 归一化为强制项。

真实样例：
```json
[
  {
    "video_id": "拥堵_20231106162147_突发事件_35056336",
    "path": "/root/autodl-tmp/高架桥数据/video_fixed/拥堵_20231106162147_突发事件_35056336.mp4",
    "stream": {
      "index": 0,
      "width": 2560,
      "height": 1440,
      "avg_frame_rate": "25/1",
      "time_base": "1/90000",
      "start_time": "0.000000",
      "duration": "60.040000",
      "nb_frames": "1501",
      "disposition": {
        "default": 1,
        "dub": 0,
        "original": 0,
        "comment": 0,
        "lyrics": 0,
        "karaoke": 0,
        "forced": 0,
        "hearing_impaired": 0,
        "visual_impaired": 0,
        "clean_effects": 0,
        "attached_pic": 0,
        "timed_thumbnails": 0
      },
      "tags": {
        "language": "und",
        "handler_name": "VideoHandler",
        "vendor_id": "[0][0][0][0]"
      }
    },
    "container_duration": "60.040000",
    "decoder_diagnostics": "",
    "first_frame_pts": 0,
    "last_frame_pts": 5400000,
    "time_base": "1/90000",
    "duration_sec": 60.0,
    "min_time_sec": 0.0,
    "fps_used": 25.0,
    "flags": [
      "dur_mismatch"
    ]
  },
  {
    "video_id": "二轮车辆闯入_20230721000905_突发事件_33392891",
    "path": "/root/autodl-tmp/高架桥数据/video_fixed/二轮车辆闯入_20230721000905_突发事件_33392891.mp4",
    "stream": {
      "index": 0,
      "width": 2560,
      "height": 1440,
      "avg_frame_rate": "25/1",
      "time_base": "1/90000",
      "start_time": "1.000000",
      "duration": "28.960000",
      "nb_frames": "724",
      "disposition": {
        "default": 1,
        "dub": 0,
        "original": 0,
        "comment": 0,
        "lyrics": 0,
        "karaoke": 0,
        "forced": 0,
        "hearing_impaired": 0,
        "visual_impaired": 0,
        "clean_effects": 0,
        "attached_pic": 0,
        "timed_thumbnails": 0
      },
      "tags": {
        "language": "und",
        "handler_name": "VideoHandler",
        "vendor_id": "[0][0][0][0]"
      }
    },
    "container_duration": "29.960000",
    "decoder_diagnostics": "",
    "first_frame_pts": 90000,
    "last_frame_pts": 2692800,
    "time_base": "1/90000",
    "duration_sec": 28.92,
    "min_time_sec": 0.0,
    "fps_used": 25.0,
    "flags": [
      "nonzero_start",
      "dur_mismatch"
    ]
  },
  {
    "video_id": "二轮车辆闯入_20230718093736_突发事件_1959790",
    "path": "/root/autodl-tmp/高架桥数据/video_fixed/二轮车辆闯入_20230718093736_突发事件_1959790.mp4",
    "stream": {
      "index": 0,
      "width": 2560,
      "height": 1440,
      "avg_frame_rate": "25/1",
      "time_base": "1/90000",
      "start_time": "1.000000",
      "duration": "29.040000",
      "nb_frames": "726",
      "disposition": {
        "default": 1,
        "dub": 0,
        "original": 0,
        "comment": 0,
        "lyrics": 0,
        "karaoke": 0,
        "forced": 0,
        "hearing_impaired": 0,
        "visual_impaired": 0,
        "clean_effects": 0,
        "attached_pic": 0,
        "timed_thumbnails": 0
      },
      "tags": {
        "language": "und",
        "handler_name": "VideoHandler",
        "vendor_id": "[0][0][0][0]"
      }
    },
    "container_duration": "30.040000",
    "decoder_diagnostics": "",
    "first_frame_pts": 90000,
    "last_frame_pts": 2700000,
    "time_base": "1/90000",
    "duration_sec": 29.0,
    "min_time_sec": 0.0,
    "fps_used": 25.0,
    "flags": [
      "nonzero_start",
      "dur_mismatch"
    ]
  }
]
```
