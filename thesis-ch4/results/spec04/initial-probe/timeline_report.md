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
