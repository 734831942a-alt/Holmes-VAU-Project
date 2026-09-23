# SPEC-04 · 全量解码帧数与媒体头差异

[
  {
    "video_id": "二轮车辆闯入_20230718183416_突发事件_1965704",
    "decoded_frames": 372,
    "nb_frames": "373"
  },
  {
    "video_id": "二轮车辆闯入_20230721111238_突发事件_33393522",
    "decoded_frames": 367,
    "nb_frames": "368"
  }
]

manifest 使用实际解码帧的 PTS；不使用 nb_frames/fps 补帧或替代时长。B0 对所选 test 样本验证 Decord 帧数一致性，不一致立即停止另记 issue。
