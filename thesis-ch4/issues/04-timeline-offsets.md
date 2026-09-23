# SPEC-04 · 时间轴差异超过探查阈值

# SPEC-04 时间轴报告

首帧归零 + 秒：t_norm(frame) = (pts(frame) − first_frame_pts) × time_base。
duration_sec 严格取最后一帧 t_norm；不加一帧时长，不使用 container duration、nb_frames/fps 或 DTS 替代。
不一致按有理数精确比较；因此常见的一帧显示时长差也计入 dur_mismatch。
当前 FFmpeg 4.4 的解码帧 PTS 字段为 pkt_pts（配置 frame_pts_field）；不使用 pkt_dts 或 best_effort_timestamp 替代。

统计范围：全量解码视频；共 6217，成功 6217，错误 0。

```json
{
  "affected_counts": {
    "nonzero_start": 5331,
    "dur_mismatch": 6217
  },
  "ratios": {
    "nonzero_start": 0.8574875341804729,
    "dur_mismatch": 1.0
  }
}
```

问题比例 >10%；§5.1 归一化为强制项。

已记录 flags；继续实现冻结的首帧归零规则，不修改公式。
