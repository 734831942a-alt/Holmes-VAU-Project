"""SPEC-04 §3 prerequisite probe, with evidence and explicit issue reporting."""
import argparse
import collections
import concurrent.futures
import json
import random
import statistics
from pathlib import Path

from data.common import config, files, read_jsonl, append_jsonl, write_text, issue
from data.timeline import probe
from data.ocr import recognize


def distribution(values):
    return {"n": len(values), "min": min(values) if values else None,
            "median": statistics.median(values) if values else None,
            "max": max(values) if values else None}


def scan(paths, cfg, destination, decode):
    cached = {r["video_id"]: r for r in read_jsonl(destination)}
    mismatched = [str(p) for p in paths if p.stem in cached and cached[p.stem]["path"] != str(p)]
    if mismatched:
        issue(cfg, "cache-source-conflict", "输入路径与缓存不一致", json.dumps(mismatched, ensure_ascii=False) + "\n须归档旧缓存后重新探查，不能把旧视频时间轴用于新根目录。")
        raise ValueError("Cached source path changed; archive obsolete cache")
    pending = [p for p in paths if p.stem not in cached]
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        futures = {pool.submit(probe, p, cfg, decode): p for p in pending}
        for future in concurrent.futures.as_completed(futures):
            path = futures[future]
            try:
                row = future.result()
                if row.get("decoder_diagnostics"):
                    issue(cfg, "decoder-diagnostics", "解码器报告媒体错误", "ffprobe 返回码为零，仍提供解码帧 PTS；保留诊断并按实际解码的首末帧计算。\n不修复原始视频，不以容器跨度替代。所有诊断逐文件保存在 timeline/header records。\n" + json.dumps({k: v for k, v in row.items() if k != "pts"}, ensure_ascii=False, indent=2))
            except Exception as exc:
                row = {"video_id": path.stem, "path": str(path), "error": str(exc)}
                issue(cfg, "timeline-decode", "解码时间轴异常", json.dumps(row, ensure_ascii=False, indent=2))
            append_jsonl(destination, row)
            cached[path.stem] = row
            if len(cached) % cfg["progress_every"] == 0:
                print("scan", destination, len(cached), "/", len(paths), flush=True)
    return [cached[p.stem] for p in paths]


def timeline_report(rows, cfg, scope):
    valid = [r for r in rows if "error" not in r]
    counts = {flag: sum(flag in r["flags"] for r in valid) for flag in ["nonzero_start", "dur_mismatch"]}
    ratios = {flag: count / len(valid) if valid else None for flag, count in counts.items()}
    examples = [{k: v for k, v in r.items() if k != "pts"} for r in valid[:cfg["example_count"]]]
    text = "# SPEC-04 时间轴报告\n\n首帧归零 + 秒：t_norm(frame) = (pts(frame) − first_frame_pts) × time_base。\n"
    text += "duration_sec 严格取最后一帧 t_norm；不加一帧时长，不使用 container duration、nb_frames/fps 或 DTS 替代。\n"
    text += "不一致按有理数精确比较；因此常见的一帧显示时长差也计入 dur_mismatch。\n"
    text += "当前 FFmpeg 4.4 的解码帧 PTS 字段为 pkt_pts（配置 frame_pts_field）；不使用 pkt_dts 或 best_effort_timestamp 替代。\n"
    text += f"\n统计范围：{scope}；共 {len(rows)}，成功 {len(valid)}，错误 {len(rows)-len(valid)}。\n"
    text += "\n```json\n" + json.dumps({"affected_counts": counts, "ratios": ratios}, ensure_ascii=False, indent=2) + "\n```\n"
    if any(r is not None and r > cfg["significant_ratio"] for r in ratios.values()):
        text += "\n问题比例 >10%；§5.1 归一化为强制项。\n"
        issue(cfg, "timeline-offsets", "时间轴差异超过探查阈值", text + "\n已记录 flags；继续实现冻结的首帧归零规则，不修改公式。")
    text += "\n真实样例：\n```json\n" + json.dumps(examples, ensure_ascii=False, indent=2) + "\n```\n"
    write_text(cfg["timeline_report"], text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = config(args.config)
    paths = files(cfg)
    counts = dict(collections.Counter(p.stem.split(cfg["category_separator"])[0] for p in paths))
    matches = len(paths) == cfg["baseline_count"] and counts == cfg["baseline_categories"]
    if not matches:
        issue(cfg, "count-mismatch", "文件数或六类分布与基线不符", json.dumps(counts, ensure_ascii=False) + f"\n总数 {len(paths)}；不改变盘点口径。")
    write_text(cfg["probe_report"], "# SPEC-04 前置探查进行中\n\n" + json.dumps({"count": len(paths), "categories": counts, "baseline_matches": matches}, ensure_ascii=False, indent=2))
    sample = random.Random(cfg["seed"]).sample(paths, cfg["probe_samples"])
    rows = scan(sample, cfg, cfg["probe_records"], True)
    report = timeline_report(rows, cfg, "§3 随机抽样")
    headers = scan(paths, cfg, cfg["header_records"], False)
    durations = [float(r["container_duration"]) for r in headers if "error" not in r and r["container_duration"] is not None]
    ocr_rows = []
    for p in sample:
        row = recognize(p, cfg, Path(cfg["probe_frames"]) / (p.stem + ".png"))
        append_jsonl(cfg["ocr_records"], row)
        ocr_rows.append(row)
    confident = sum(r["eligible"] for r in ocr_rows)
    if not confident:
        issue(cfg, "ocr-unverified", "抽样 ROI 无高置信度机位识别", "不能把低置信度 OCR 当作相机 ID；所有不确定样本进入 UNKNOWN_i 和 uncertain，等待人工复核。\n无高置信度结果不能证明画面没有叠字。")
    text = "# SPEC-04 自采前置探查\n\n输入：" + cfg["src"] + "\n"
    text += "选择画像明确对应的 video_fixed 集合；不把 video/ 原片与派生副本相加。原始数据只读。\n"
    text += "\n## A 数据画像复核\n```json\n" + json.dumps({"file_count": len(paths), "categories": counts, "baseline_matches": matches, "container_duration_distribution_sec": distribution(durations), "over120_count": sum(d > cfg["long_threshold_sec"] for d in durations), "baseline_container_duration": cfg["baseline_container_duration"], "header_errors": [r for r in headers if "error" in r]}, ensure_ascii=False, indent=2) + "\n```\n"
    text += "与《数据集画像_2026-09-19》的文件数/六类分布比较如上；容器跨度统计仅用于画像对照，manifest 使用解码时间轴。\n"
    text += f"抽样 {len(ocr_rows)} 帧；高置信度机位 OCR {confident}；固定 ROI、原始 OCR 和置信度见 {cfg['ocr_records']}，裁剪帧见 {cfg['probe_frames']}。\n"
    text += "有可 OCR 叠字样例。低置信度/不完整叠字不能据此猜测相机。\n" if confident else "机位叠字尚未由高置信度 OCR 验证，已记 issue。\n"
    text += "\n## B 时间轴探查\n" + report
    write_text(cfg["probe_report"], text)
    print("probe complete", flush=True)


if __name__ == "__main__":
    main()
