"""
analyze_v4g_debug.py  —  逐条分析 postprocess_lcrm_lite_fusion.py 的输出，
重点列出 FP（GT≠多车事故 but pred=多车事故）和 FN（GT=多车事故 but pred≠多车事故）。

用法：
  python analyze_v4g_debug.py \
      --pred /root/autodl-tmp/高架桥数据/traffic_test_pred_e8_lcrm_lite_smoke_240_v4g.jsonl \
      [--gt-label 多车事故] [--output-fp fp_debug.jsonl] [--output-fn fn_debug.jsonl]
"""
import argparse
import json
from pathlib import Path


MULTICAR = "多车事故"


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def label_from_video(video):
    return str(video or "").split("_")[0]


def fmt(v, width=8):
    if v is None:
        return " " * width
    if isinstance(v, float):
        return f"{v:.3f}".rjust(width)
    return str(v).rjust(width)


def short_debug(d):
    """Extract the most diagnostic fields from lcrm_lite_debug."""
    if not isinstance(d, dict):
        return {}
    bp = d.get("best_pair") or {}
    return {
        "reason": d.get("reason", ""),
        "channel": d.get("channel"),
        "sc": d.get("static_count"),
        "score": bp.get("score"),
        "compact": bp.get("compactness"),
        "persist": bp.get("persistence_frames"),
        "person_pk": bp.get("person_peak"),
        "person_bonus": bp.get("person_bonus"),
        "rel_compact": bp.get("relative_compactness"),
        "congestion_acc": d.get("congestion_accident_channel"),
        "construction_bypass": d.get("construction_bypass"),
        "area_retry": d.get("area_relaxed_retry"),
    }


def print_table(rows, title):
    if not rows:
        print(f"\n{'='*70}\n{title} : 0 条\n")
        return
    print(f"\n{'='*70}")
    print(f"{title} : {len(rows)} 条")
    print(f"{'='*70}")
    hdr = (
        f"{'video':30s}  {'GT':8s}  {'base':8s}  {'fused':8s}  "
        f"{'sc':>4}  {'score':>6}  {'cmpct':>5}  {'pers':>4}  "
        f"{'p_pk':>4}  {'p_bon':>5}  {'reason'}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        d = short_debug(r.get("lcrm_lite_debug"))
        vid = str(r.get("video", ""))[-30:]
        print(
            f"{vid:30s}  "
            f"{str(r['_gt']):8s}  "
            f"{str(r.get('lcrm_lite_base_label') or ''):8s}  "
            f"{str(r.get('lcrm_lite_fused_label') or ''):8s}  "
            f"{fmt(d.get('sc'), 4)}  "
            f"{fmt(d.get('score'), 6)}  "
            f"{fmt(d.get('compact'), 5)}  "
            f"{fmt(d.get('persist'), 4)}  "
            f"{fmt(d.get('person_pk'), 4)}  "
            f"{fmt(d.get('person_bonus'), 5)}  "
            f"{d.get('reason', '')}"
            + (f"  [cac={d.get('congestion_acc')}]" if d.get("congestion_acc") else "")
            + (f"  [bypass]" if d.get("construction_bypass") else "")
            + (f"  [area_retry]" if d.get("area_retry") else "")
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="v4g output jsonl")
    ap.add_argument("--gt-label", default=MULTICAR)
    ap.add_argument("--output-fp", default="", help="dump FP rows to jsonl")
    ap.add_argument("--output-fn", default="", help="dump FN rows to jsonl")
    args = ap.parse_args()

    rows = read_jsonl(args.pred)
    for r in rows:
        r["_gt"] = label_from_video(r.get("video", ""))
        r["_pred_label"] = r.get("lcrm_lite_fused_label") or r.get("lcrm_lite_base_label")

    target = args.gt_label

    tp = [r for r in rows if r["_gt"] == target and r["_pred_label"] == target]
    fp = [r for r in rows if r["_gt"] != target and r["_pred_label"] == target]
    fn = [r for r in rows if r["_gt"] == target and r["_pred_label"] != target]
    tn = [r for r in rows if r["_gt"] != target and r["_pred_label"] != target]

    prec = len(tp) / max(1, len(tp) + len(fp))
    rec  = len(tp) / max(1, len(tp) + len(fn))
    f1   = 2 * prec * rec / max(1e-9, prec + rec)

    print(f"\n>>> 目标类: {target}")
    print(f"    TP={len(tp)}  FP={len(fp)}  FN={len(fn)}  TN={len(tn)}")
    print(f"    Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")

    # ── FP breakdown by GT category ──────────────────────────────────
    print(f"\n--- FP by GT category ---")
    from collections import Counter
    fp_by_gt = Counter(r["_gt"] for r in fp)
    for gt, cnt in sorted(fp_by_gt.items(), key=lambda x: -x[1]):
        print(f"  GT={gt}: {cnt}")

    print_table(fp, f"FP 详情 (GT≠{target} → pred={target})")
    print_table(fn, f"FN 详情 (GT={target} → pred≠{target})")

    # ── TP: what channels fired ───────────────────────────────────────
    print(f"\n--- TP channel breakdown ---")
    tp_channels = Counter(
        (short_debug(r.get("lcrm_lite_debug")).get("channel") or "congestion_acc")
        for r in tp
    )
    for ch, cnt in tp_channels.most_common():
        print(f"  {ch}: {cnt}")

    # ── FN reason breakdown ───────────────────────────────────────────
    print(f"\n--- FN reason breakdown ---")
    fn_reasons = Counter(short_debug(r.get("lcrm_lite_debug")).get("reason") for r in fn)
    for reason, cnt in fn_reasons.most_common():
        print(f"  {reason}: {cnt}")

    if args.output_fp:
        Path(args.output_fp).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in fp), encoding="utf-8"
        )
        print(f"\nFP rows -> {args.output_fp}")
    if args.output_fn:
        Path(args.output_fn).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in fn), encoding="utf-8"
        )
        print(f"FN rows -> {args.output_fn}")


if __name__ == "__main__":
    main()
