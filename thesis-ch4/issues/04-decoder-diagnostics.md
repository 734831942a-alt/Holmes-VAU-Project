# SPEC-04 · 全量媒体解码诊断

全量 6217 段中有 134 段 ffprobe 返回零退出码但报告媒体解码诊断。所有文件均提供满足首帧归零/末帧正时长的实际解码 PTS。

逐文件完整诊断见 results/spec04/media-anomalies.json 与 timeline.jsonl。manifest 对这些文件保留 decode_warning。未修改原片、未删除样本、未替换为容器时长。
