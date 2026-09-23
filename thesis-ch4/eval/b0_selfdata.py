"""Single-attempt greedy B0. Failed parsing is persisted, never retried or corrected."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics

from data.common import config, read_jsonl, append_jsonl, write_json, write_jsonl, fingerprint, issue
from data.timeline import normalize_pts


def prediction(video_id, raw_output, frames_seen, cfg):
    matches = list(re.finditer(cfg["parse_regex"], raw_output))
    match = matches[0] if len(matches) == 1 else None
    start = float(match.group("start")) if match else None
    end = float(match.group("end")) if match else None
    ok = bool(match and math.isfinite(start) and math.isfinite(end) and 0 <= start < end)
    # Parsed numbers are kept as emitted. No clamping, offset correction, retry or filtering.
    return {"video_id": video_id, "raw_output": raw_output, "parsed_start": start,
            "parsed_end": end, "parse_ok": ok, "frames_seen": frames_seen,
            "format_ok": bool(re.fullmatch(cfg["format_regex"], raw_output)),
            "failure_reason": None if ok else "parse_failure"}


def summarize(values, cfg):
    import numpy as np
    return {"count": len(values), "min": min(values) if values else None,
            "median": statistics.median(values) if values else None,
            "max": max(values) if values else None,
            "quantiles": {str(q): float(np.quantile(values, q)) if values else None
                          for q in cfg["distribution_quantiles"]}}


def metrics(records, manifest, cfg):
    by_id = {r["video_id"]: r for r in manifest}
    valid = [r for r in records if r["parse_ok"]]
    n = len(records)
    full = sum(abs(r["parsed_start"]) <= cfg["full_clip_tolerance_sec"] and
               abs(r["parsed_end"] - by_id[r["video_id"]]["duration_sec"]) <= cfg["full_clip_tolerance_sec"]
               for r in valid)
    return {"sample_count": n, "expected_count": len(manifest), "complete": n == len(manifest),
            "parse_fail_rate": (n - len(valid)) / n if n else None,
            "format_ok_rate": sum(r["format_ok"] for r in records) / n if n else None,
            "duration_distribution_sec": summarize([r["parsed_end"] - r["parsed_start"] for r in valid], cfg),
            "start_distribution_sec": summarize([r["parsed_start"] for r in valid], cfg),
            "avg_frames": statistics.mean(r["frames_seen"] for r in records) if n else None,
            "full_clip_ratio": full / n if n else None,
            "full_clip_tolerance_sec": cfg["full_clip_tolerance_sec"],
            "rate_denominator": "all persisted test records; distributions use parse_ok records",
            "interval_out_of_bounds_count": sum(r["parsed_end"] > by_id[r["video_id"]]["duration_sec"] for r in valid),
            "parse_semantics": "Exactly one START/END pair, finite and 0 <= start < end; no clipping to video bounds",
            "format_semantics": "Exact full output match to configured format, including whitespace",
            "pilot_enabled": cfg["pilot_enabled"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--a1-config", help="Required for future runs; legacy B0 outputs remain immutable")
    args = parser.parse_args()
    cfg = config(args.config)
    if not args.a1_config:
        raise RuntimeError("Future runs require --a1-config for parsing and authoritative generation evidence; no generation started")
    from eval.reparse_v2 import IntervalParser, measurement
    ac = config(args.a1_config)
    if cfg["max_new_tokens"] != ac["max_new_tokens"]:
        raise RuntimeError("A1 generation budget conflict")
    if Path(args.out).resolve() == Path(ac['original_predictions']).resolve().parent:
        raise RuntimeError("Original B0 output directory is immutable; do not resume or overwrite")
    interval_parser = IntervalParser(ac, cfg['format_regex'])
    dc = config(cfg["data_config"])
    manifest = read_jsonl(args.manifest)
    splits = json.loads(Path(dc["splits"]).read_text())
    from data.make_splits import leakage
    if any(leakage(manifest, splits, dc).values()):
        raise RuntimeError("Split leakage hard gate")
    rows = [r for r in manifest if r["video_id"] in set(splits[args.split])]
    if not rows or any(r["split"] != args.split for r in rows):
        raise RuntimeError("Empty/inconsistent requested split")
    future_cfg = dict(ac, expected_test=len(rows))
    by_id = {r['video_id']: r for r in rows}
    def current_metrics(records):
        return {**measurement(records, by_id, future_cfg), 'n_attempted': len(attempts),
                'n_persisted': len(records), 'n_interrupted': len(attempts) - len(records)}
    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    pred_path, run_path = destination / cfg["predictions_file"], destination / cfg["run_file"]
    attempt_path = destination / cfg["attempts_file"]
    run = {"config_sha256": fingerprint(cfg), "manifest_sha256": fingerprint(manifest), "split": args.split,
           "model": cfg["model_path"], "prompt_template": cfg["prompt_template"], "data_config_sha256": fingerprint(dc)}
    run['a1_config_sha256'] = fingerprint(ac)
    if run_path.exists():
        if not args.resume or json.loads(run_path.read_text()) != run:
            raise RuntimeError("Existing B0 run must resume with identical config, prompt, manifest and split")
    else:
        if pred_path.exists() or attempt_path.exists():
            raise RuntimeError("Existing unverified inference artifacts; refusing overwrite")
        write_json(run_path, run)
    records = read_jsonl(pred_path)
    done = {r["video_id"] for r in records}
    attempts = read_jsonl(attempt_path)
    attempted = {r["video_id"] for r in attempts}
    if len(done) != len(records) or len(attempted) != len(attempts) or not done <= {r["video_id"] for r in rows}:
        raise RuntimeError("Duplicate or foreign inference records")
    if attempted != done:
        issue(dc, "b0-interrupted", "B0 已尝试但未落盘完成", str(sorted(attempted.symmetric_difference(done))) + "\n禁止自动重试，等待人工确认运行状态。")
        raise RuntimeError("Unresolved inference attempt; no retry")
    pending = [r for r in rows if r["video_id"] not in done]
    write_json(destination / cfg["metrics_file"], current_metrics(records))
    if not pending:
        write_json(destination / cfg["metrics_file"], current_metrics(records))
        return
    import torch
    from decord import VideoReader, cpu
    from src.model.holmesvau_infer import load_model, uniform_indices
    from eval.generation_evidence import generate_with_evidence
    torch.set_num_threads(cfg["torch_threads"])
    timeline = {r["video_id"]: r for r in read_jsonl(dc["timeline_records"])}
    try:
        model, tokenizer = load_model(cfg["model_path"], cfg["device"])
    except Exception as exc:
        issue(dc, "model-load", "冻结模型加载失败", repr(exc) + "\n未修改冻结推理模块，不升级受保护包。")
        raise
    for row in pending:
        proof = timeline[row["video_id"]]
        times = normalize_pts(proof["pts"], proof["time_base"])
        reader = VideoReader(row["path"], ctx=cpu(dc["stream_index"]), num_threads=dc["decode_threads"])
        if len(reader) != len(times):
            issue(dc, "decoder-frame-mismatch", "模型解码器与 ffprobe 帧数不同", row["video_id"] + f"\nDecord={len(reader)}; ffprobe={len(times)}。无法保证采样帧对应同一时间轴，停止 B0。")
            raise RuntimeError("Decoder frame-count conflict")
        indices = uniform_indices(len(reader), cfg["num_frames"])
        del reader
        frame_times = cfg["frame_time_separator"].join(cfg["frame_time_template"].format(index=i, time_sec=float(times[index]))
                                                     for i, index in enumerate(indices, start=1))
        prompt = cfg["prompt_template"].format(category=row["category"], duration_sec=row["duration_sec"], frame_times=frame_times)
        attempt = {"video_id": row["video_id"], "prompt": prompt, "frame_indices": indices,
                   "status": "started", "stop_reason": "unknown", "generated_token_count": None,
                   "generated_token_ids": None, "last_token_is_eos": None,
                   "max_new_tokens": cfg["max_new_tokens"], "truncated": None}
        append_jsonl(attempt_path, attempt)
        attempts.append(attempt)
        # Any inference exception stops here with the durable attempt ledger. Resume cannot retry it.
        try:
            raw, actual_indices, evidence = generate_with_evidence(row["path"], prompt, model, tokenizer,
                                           max_new_tokens=cfg["max_new_tokens"], temperature=cfg["temperature"],
                                           num_frames=cfg["num_frames"], input_size=cfg["input_size"],
                                           max_tiles_per_frame=cfg["max_tiles_per_frame"])
        except Exception as exc:
            issue(dc, "b0-inference", "B0 推理异常，已停止且不重试", row["video_id"] + "\n" + repr(exc) + "\n调用前 attempt 已持久化，不将基础设施失败伪装成解析失败。")
            raise
        # Persist even when a subsequent frame-alignment check rejects the result.
        attempt.update(evidence, status="generated")
        write_jsonl(attempt_path, attempts)
        if actual_indices != indices:
            issue(dc, "model-frame-mismatch", "推理实际采样帧不一致", row["video_id"])
            raise RuntimeError("Actual sampled frames differ from supplied frame times")
        # Persist actual token evidence before the parsed record; a crash between
        # these writes remains an interrupted attempt and cannot trigger a retry.
        result = {"video_id": row["video_id"], "raw_output": raw, "frames_seen": len(actual_indices),
                  **interval_parser.parse(raw, row['duration_sec']), **evidence}
        append_jsonl(pred_path, result)
        records.append(result)
        write_json(destination / cfg["metrics_file"], current_metrics(records))
        monitor = ac['future_monitor']
        if len(records) == monitor['first_batch'] or len(records) % monitor['every'] == 0:
            recent = records[-monitor['every']:]
            check = {'n': len(recent), 'parse_fail_rate': sum(not r['parse_ok'] for r in recent) / len(recent),
                     'format_fail_rate': sum(not r['format_ok'] for r in recent) / len(recent),
                     'truncated_rate': sum(r['truncated'] for r in recent) / len(recent), 'last_raw_output': raw}
            print('B0 checkpoint', json.dumps(check, ensure_ascii=False), flush=True)
            if (check['parse_fail_rate'] >= monitor['parse_fail_pause_rate'] or
                check['format_fail_rate'] >= monitor['format_fail_pause_rate'] or
                check['truncated_rate'] >= monitor['truncated_pause_rate']):
                issue(dc, 'a1-monitor-pause', 'B0 中途检查暂停，需先检查原文与生成证据', json.dumps(check, ensure_ascii=False))
                raise RuntimeError('B0 monitoring pause; do not resume without reviewing evidence')
        if len(records) % cfg["progress_every"] == 0:
            print("B0", len(records), "/", len(rows), flush=True)
    write_json(destination / cfg["metrics_file"], current_metrics(records))
    print("B0 complete", len(records), flush=True)


if __name__ == "__main__":
    main()
