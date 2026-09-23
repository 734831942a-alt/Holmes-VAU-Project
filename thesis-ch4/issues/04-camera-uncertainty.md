# SPEC-04 · 机位分组需要人工复核

{
  "uncertain_count": 5539,
  "uncertain_reasons": {
    "cluster_boundary": 2881,
    "low_confidence_or_missing_overlay": 2658
  },
  "known_group_count": 189
}

这些样本严格按 §5.2 标为各自 UNKNOWN_i，原始 OCR、置信度、ROI、帧号和原因保存在 data/camera_groups.json 的 uncertain 中。
未调整阈值来强行增加确定相机数，未臆造相机 ID。cross_split_leak=0 证明输出的 camera_group 不跨 split，不证明未复核 UNKNOWN 样本在现实中属于不同相机；需人工复核这些机位。
已生成 test 按 §5.3 锁定，不因结果或人工猜测自动重抽。
