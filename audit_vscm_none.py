"""
audit_vscm_none.py
------------------
离线分析 OVD summary jsonl 中多车事故样本的 VSCM=None 失败原因。
不需要 GPU，只需要已有的 OVD v7 summary jsonl 文件。

用法（服务器）：
  python audit_vscm_none.py \
    --ovd-jsonl /root/autodl-tmp/高架桥数据/traffic_train_ovd_summary_v7.jsonl \
    --target-label 多车事故

可选：同时指定 --all-labels 打印所有类别的 VSCM 触发率对比
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


FAILURE_ORDER = [
    # early filter (vscm_debug)
    "video_not_found",
    "ovd_not_ok",
    "no_frame_or_invalid_size",
    "total_count_lt_5",
    "static_count_out_2_6",
    "static_ratio_gt_0.28",
    "moving_count_lt_5",
    "cluster_center_out_0.12_0.88",
    # passed candidate, not triggered (proximity_reason)
    "candidate:no_relation_no_context",
    "candidate:context_without_vehicle_relation",
    "candidate:relation_without_scene_context",
    "candidate:queue_like_suppressed",
    # triggered but motorcycle suppressed
    "suppressed_motorcycle",
    # success
    "TRIGGERED",
]


def extract_label(video: str) -> str:
    """Label = prefix before first underscore in basename."""
    if not video:
        return ""
    basename = Path(video).name
    if "_" in basename:
        return basename.split("_", 1)[0].strip()
    return basename.strip()


def classify_sample(rec: dict) -> str:
    """Return one of the FAILURE_ORDER strings."""
    if not rec.get("ok"):
        err = rec.get("err", "")
        if err == "video_not_found":
            return "video_not_found"
        return "ovd_not_ok"

    ovd = rec.get("ovd") or {}

    # Early filter fail
    vscm_debug = ovd.get("vscm_debug")
    if vscm_debug and vscm_debug.get("debug_only"):
        return vscm_debug.get("none_reason", "unknown_early_filter")

    # Passed candidate generation
    vscm = ovd.get("vscm")
    if not vscm:
        # Neither vscm nor vscm_debug — shouldn't happen but guard it
        return "ovd_not_ok"

    if vscm.get("triggered", False):
        return "TRIGGERED"

    # Motorcycle suppression
    if vscm.get("suppression_reason") == "motorcycle_evidence":
        return "suppressed_motorcycle"

    # Not triggered: use proximity_reason
    pr = vscm.get("proximity_reason")
    if pr is None:
        return "candidate:no_relation_no_context"
    return f"candidate:{pr}"


def print_table(label: str, category_counts: dict, total: int):
    print(f"\n{'='*64}")
    print(f"  类别: {label}  (共 {total} 条)")
    print(f"{'='*64}")
    print(f"  {'失败原因':<42} {'数量':>5}  {'占比':>6}")
    print(f"  {'-'*56}")

    ordered = []
    for k in FAILURE_ORDER:
        if k in category_counts:
            ordered.append((k, category_counts[k]))
    # Any unexpected keys
    for k, v in category_counts.items():
        if k not in FAILURE_ORDER:
            ordered.append((k, v))

    for reason, cnt in ordered:
        pct = cnt / max(total, 1) * 100
        marker = " ✓" if reason == "TRIGGERED" else ""
        print(f"  {reason:<42} {cnt:>5}  {pct:>5.1f}%{marker}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ovd-jsonl", required=True, help="Path to OVD summary jsonl (e.g. traffic_train_ovd_summary_v7.jsonl)")
    ap.add_argument("--target-label", default="多车事故", help="Primary label to audit in detail")
    ap.add_argument("--all-labels", action="store_true", help="Print trigger rate for all labels")
    args = ap.parse_args()

    path = Path(args.ovd_jsonl)
    if not path.exists():
        print(f"[ERROR] file not found: {path}")
        return

    # label -> list of (reason, detail_dict)
    label_buckets: dict[str, Counter] = defaultdict(Counter)
    label_totals: dict[str, int] = defaultdict(int)

    # For detailed inspection of target label
    detail_rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            video = rec.get("video", "")
            label = extract_label(video)
            reason = classify_sample(rec)

            label_buckets[label][reason] += 1
            label_totals[label] += 1

            if label == args.target_label:
                # Collect extra stats for detail report
                ovd = (rec.get("ovd") or {})
                vscm_debug = ovd.get("vscm_debug") or {}
                vscm = ovd.get("vscm") or {}
                detail_rows.append({
                    "video": video,
                    "reason": reason,
                    # from debug or vscm
                    "total_count": vscm_debug.get("total_count") or vscm.get("static_count", 0) + vscm.get("moving_count", 0),
                    "static_count": vscm_debug.get("static_count") or vscm.get("static_count"),
                    "moving_count": vscm_debug.get("moving_count") or vscm.get("moving_count"),
                    "static_ratio": vscm_debug.get("static_ratio") or vscm.get("static_ratio"),
                    "cluster_center_x_ratio": vscm_debug.get("cluster_center_x_ratio") or vscm.get("cluster_center_x_ratio"),
                    "context_frames": vscm.get("context_frames"),
                    "contact_frames": vscm.get("contact_frames"),
                    "close_frames": vscm.get("close_frames"),
                    "context_hits": vscm.get("context_hits"),
                    "meaningful_scene_context": vscm.get("meaningful_scene_context"),
                    "queue_like": vscm.get("queue_like"),
                })

    # ── Detailed table for target label ──────────────────────────────────────
    target = args.target_label
    target_total = label_totals.get(target, 0)
    if target_total == 0:
        print(f"[WARN] No samples found for label '{target}'. Check --target-label.")
    else:
        print_table(target, label_buckets[target], target_total)

    # ── Failure-mode detail stats for candidate rows ──────────────────────────
    # For samples that passed early filter, show distribution of numeric stats
    candidate_rows = [r for r in detail_rows if r["reason"].startswith("candidate:") or r["reason"] == "TRIGGERED" or r["reason"] == "suppressed_motorcycle"]
    if candidate_rows:
        print(f"  ── 通过候选阶段的 {target} 样本 (n={len(candidate_rows)}) 数值统计 ──")
        print(f"  {'reason':<44} {'static':>6} {'moving':>7} {'s_ratio':>8} {'ctx_f':>6} {'cont_f':>7}")
        print(f"  {'-'*80}")
        for r in candidate_rows:
            sc = r["static_count"] if r["static_count"] is not None else "-"
            mc = r["moving_count"] if r["moving_count"] is not None else "-"
            sr = f"{r['static_ratio']:.2f}" if r["static_ratio"] is not None else "-"
            cf = r["context_frames"] if r["context_frames"] is not None else "-"
            co = r["contact_frames"] if r["contact_frames"] is not None else "-"
            short_reason = r["reason"].replace("candidate:", "")
            print(f"  {short_reason:<44} {str(sc):>6} {str(mc):>7} {str(sr):>8} {str(cf):>6} {str(co):>7}")

    # ── Early-filter detail: total_count distribution ────────────────────────
    early_total_lt5 = [r for r in detail_rows if r["reason"] == "total_count_lt_5"]
    if early_total_lt5:
        counts = [r["total_count"] for r in early_total_lt5 if r["total_count"] is not None]
        if counts:
            from collections import Counter as Ct
            dist = Ct(counts)
            print(f"\n  ── total_count_lt_5 分布 (n={len(early_total_lt5)}) ──")
            for v in sorted(dist):
                print(f"    total_count={v}: {dist[v]}条")

    # ── All-label trigger rate comparison ─────────────────────────────────────
    if args.all_labels:
        print(f"\n{'='*64}")
        print("  所有类别 VSCM 触发率对比")
        print(f"{'='*64}")
        print(f"  {'类别':<16} {'总数':>6} {'触发':>6} {'触发率':>8} {'候选数':>8} {'候选内触发率':>12}")
        print(f"  {'-'*60}")
        for lbl in sorted(label_totals.keys()):
            total = label_totals[lbl]
            triggered = label_buckets[lbl].get("TRIGGERED", 0)
            # candidate = passed early filter
            early_fail = sum(
                v for k, v in label_buckets[lbl].items()
                if k not in ("video_not_found", "ovd_not_ok") and not k.startswith("candidate:") and k != "TRIGGERED" and k != "suppressed_motorcycle"
            )
            candidate = total - label_buckets[lbl].get("video_not_found", 0) - label_buckets[lbl].get("ovd_not_ok", 0) - early_fail
            trigger_rate = triggered / max(total, 1) * 100
            candidate_trigger_rate = triggered / max(candidate, 1) * 100
            print(f"  {lbl:<16} {total:>6} {triggered:>6} {trigger_rate:>7.1f}% {candidate:>8} {candidate_trigger_rate:>11.1f}%")


if __name__ == "__main__":
    main()
