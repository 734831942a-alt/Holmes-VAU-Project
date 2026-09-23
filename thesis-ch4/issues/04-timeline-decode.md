# SPEC-04 · 首轮 PTS 字段兼容问题（已修复）

初轮请求 pts 字段，但服务器 FFmpeg 4.4 的解码帧字段是 pkt_pts，导致 50 段均被探查器判为缺失 PTS。
配置现已显式指定 frame_pts_field: pkt_pts；不使用 DTS 或 best-effort 时间戳替代。修正后同一随机 50 段全部得到有效首帧归零时间轴及真实字段样例。

初轮证据：results/spec04/initial-probe/probe-timeline.jsonl。有效抽样证据：results/spec04/probe-timeline.jsonl。
媒体本身的解码诊断另见 issues/04-decoder-diagnostics.md；不将其删除或误称为探查器问题。
