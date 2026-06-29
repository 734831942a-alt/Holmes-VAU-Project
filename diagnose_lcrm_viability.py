import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from eval_traffic_fixed import LABELS, UNKNOWN_LABEL, detect_label, get_gt_label
from postprocess_lcrm_lite_fusion import get_ovd_payload, lcrm_lite_signal, parse_csv, read_jsonl


TARGET = "多车事故"


def by_video(rows):
    return {row.get("video"): row for row in rows if row.get("video")}


def label_or_unknown(text):
    return detect_label(text) or UNKNOWN_LABEL


def pct(num, den):
    return round(100.0 * num / den, 2) if den else 0.0


def quantiles(values):
    values = sorted(x for x in values if x is not None)
    if not values:
        return {}
    def q(p):
        idx = int(round((len(values) - 1) * p))
        return round(values[idx], 4)
    return {
        "n": len(values),
        "min": round(values[0], 4),
        "p25": q(0.25),
        "median": q(0.50),
        "p75": q(0.75),
        "max": round(values[-1], 4),
    }


def compact(counter, limit=40):
    return {str(k): v for k, v in counter.most_common(limit)}


def get_tracks(lcrm):
    if not isinstance(lcrm, dict):
        return []
    return [x for x in lcrm.get("tracks", []) if isinstance(x, dict)]


def track_stats(lcrm):
    tracks = get_tracks(lcrm)
    static_tracks = [
        t for t in tracks
        if t.get("bucket") == "static"
        and t.get("last_box")
    ]
    area_vals = []
    frame_vals = []
    step_vals = []
    for t in static_tracks:
        box = t.get("last_box") or []
        if len(box) == 4:
            area_vals.append(max(0, box[2] - box[0]) * max(0, box[3] - box[1]))
        frame_vals.append(int(t.get("frames", 0) or 0))
        step_vals.append(float(t.get("max_step_norm", 99.0) or 99.0))
    return {
        "track_count": len(tracks),
        "static_tracks": len(static_tracks),
        "static_count": int((lcrm or {}).get("static_count", 0) or 0) if isinstance(lcrm, dict) else 0,
        "frames_max": max(frame_vals) if frame_vals else None,
        "step_min": min(step_vals) if step_vals else None,
        "area_max_abs": max(area_vals) if area_vals else None,
    }


def add_example(bucket, item, limit):
    if len(bucket) < limit:
        bucket.append(item)


def main():
    ap = argparse.ArgumentParser(description="Decide whether LCRM still has separable signal worth tuning.")
    ap.add_argument("--base-pred", required=True)
    ap.add_argument("--ovd", required=True)
    ap.add_argument("--base-pred-field", default="pred")
    ap.add_argument("--gt-field", default="gt")
    ap.add_argument("--allowed-base-labels", default="异常停车", type=parse_csv)
    ap.add_argument("--max-examples", type=int, default=20)

    # Same knobs as postprocess/lcrm coverage.
    ap.add_argument("--far-y", type=float, default=0.45)
    ap.add_argument("--mid-y", type=float, default=0.75)
    ap.add_argument("--far-weight", type=float, default=0.45)
    ap.add_argument("--mid-weight", type=float, default=0.80)
    ap.add_argument("--min-area-ratio", type=float, default=0.0015)
    ap.add_argument("--max-pair-area-ratio", type=float, default=0.0)
    ap.add_argument("--close-norm-gap", type=float, default=0.35)
    ap.add_argument("--lateral-suppress-dx", type=float, default=0.40)
    ap.add_argument("--lateral-suppress-dy", type=float, default=0.20)
    ap.add_argument("--min-pair-compactness", type=float, default=0.35)
    ap.add_argument("--min-track-frames", type=int, default=3)
    ap.add_argument("--min-static-count", type=int, default=4)
    ap.add_argument("--score-thr-weak", type=float, default=0.75)
    ap.add_argument("--max-norm-gap-weak", type=float, default=0.10)
    ap.add_argument("--min-persistence-weak", type=int, default=4)
    ap.add_argument("--construction-bypass-score", type=float, default=0.0)
    ap.add_argument("--construction-bypass-score-congestion", type=float, default=None)
    ap.add_argument("--construction-bypass-static-count", type=int, default=3)
    ap.add_argument("--stationary-step", type=float, default=0.04)
    ap.add_argument("--persistence-norm", type=float, default=4.0)
    ap.add_argument("--close-weight", type=float, default=0.40)
    ap.add_argument("--persistence-weight", type=float, default=0.35)
    ap.add_argument("--stationary-weight", type=float, default=0.25)
    ap.add_argument("--score-thr", type=float, default=0.65)
    ap.add_argument("--congestion-vehicle-peak", type=int, default=12)
    ap.add_argument("--congestion-slow-ratio", type=float, default=0.70)
    ap.add_argument("--congestion-slow-regions", type=int, default=3)
    ap.add_argument("--congestion-score-penalty", type=float, default=0.10)
    ap.add_argument("--congestion-min-compactness", type=float, default=0.70)
    ap.add_argument("--congestion-min-persistence", type=int, default=8)
    ap.add_argument("--congestion-max-static", type=int, default=5)
    ap.add_argument("--congestion-band-tolerance", type=float, default=0.15)
    ap.add_argument("--congestion-min-relative-compactness", type=float, default=1.75)
    ap.add_argument("--motorcycle-peak", type=int, default=3)
    ap.add_argument("--twowheel-exempt-static-count", type=int, default=4)
    ap.add_argument("--twowheel-exempt-local-pair-raw", type=float, default=0.45)
    ap.add_argument("--twowheel-exempt-contact-frames", type=int, default=3)
    ap.add_argument("--twowheel-exempt-close-frames", type=int, default=8)
    ap.add_argument("--enable-area-relaxed-retry", action="store_true")
    ap.add_argument("--area-relaxed-ratio-mul", type=float, default=0.5)
    ap.add_argument("--area-relaxed-min-area-ratio", type=float, default=0.0004)
    ap.add_argument("--area-relaxed-min-contact-frames", type=int, default=3)
    ap.add_argument("--area-relaxed-min-close-frames", type=int, default=8)
    ap.add_argument("--person-bonus", type=float, default=0.0)
    ap.add_argument("--person-bonus-min-peak", type=int, default=3)
    ap.add_argument("--congestion-strong-rel-compact", type=float, default=99.0)
    ap.add_argument("--congestion-strong-min-sc", type=int, default=0)
    ap.add_argument("--congestion-strong-max-compact", type=float, default=1.0)
    ap.add_argument("--cone-peak", type=int, default=8)
    ap.add_argument("--construction-vehicle-peak", type=int, default=4)
    ap.add_argument("--cone-pair-peak", type=int, default=5)
    ap.add_argument("--barrier-pair-peak", type=int, default=3)
    ap.add_argument("--block-debris-context", action="store_true")
    ap.add_argument("--debris-peak", type=int, default=1)
    ap.add_argument("--road-obstacle-peak", type=int, default=3)
    args = ap.parse_args()

    rows = read_jsonl(args.base_pred)
    ovd_map = by_video(read_jsonl(args.ovd))

    # Oracle-like route stats: what if every signal were allowed for selected base labels?
    route_grid = [
        ("parking", {"异常停车"}),
        ("parking+congestion", {"异常停车", "拥堵"}),
        ("parking+congestion+unknown", {"异常停车", "拥堵", UNKNOWN_LABEL}),
        ("any_non_multicar", set(LABELS + [UNKNOWN_LABEL]) - {TARGET}),
    ]
    route_stats = {name: Counter() for name, _ in route_grid}

    score_near = defaultdict(list)
    score_all = defaultdict(list)
    best_fields = defaultdict(list)
    signal_gt = Counter()
    signal_base = Counter()
    signal_gt_base = Counter()
    signal_precision_pool = Counter()
    reason_missed_target = Counter()
    reason_false_signal = Counter()
    no_pair_stats = defaultdict(list)
    static_low_stats = defaultdict(list)

    examples = defaultdict(list)

    for row in rows:
        video = row.get("video", "")
        gt = get_gt_label(row, args.gt_field)
        base = label_or_unknown(row.get(args.base_pred_field))
        ovd = get_ovd_payload(ovd_map.get(video, {}))
        lcrm = ovd.get("lcrm") if isinstance(ovd, dict) else None
        signal, debug = lcrm_lite_signal(lcrm, args)
        reason = debug.get("reason")
        best = debug.get("best_pair") or {}

        if best:
            score = best.get("score")
            req = debug.get("required_score")
            margin = score - req if isinstance(score, (int, float)) and isinstance(req, (int, float)) else None
            key = "target_missed" if gt == TARGET and base != TARGET else ("non_target" if gt != TARGET else "target_already")
            score_all[key].append(margin)
            for f in ["score", "compactness", "edge_norm_gap", "dx_union", "dy_union", "persistence_frames", "relative_compactness"]:
                best_fields[(key, f)].append(best.get(f))
            if margin is not None and -0.15 <= margin < 0:
                score_near[(gt, base)].append((margin, video, reason, best))

        if signal:
            signal_gt[gt] += 1
            signal_base[base] += 1
            signal_gt_base[(gt, base)] += 1
            if gt == TARGET:
                signal_precision_pool["tp_signal"] += 1
            else:
                signal_precision_pool["fp_signal"] += 1
                reason_false_signal[(gt, reason)] += 1
            add_example(examples["signal"], (video, gt, base, reason, debug.get("channel"), best), args.max_examples)

        for name, allowed in route_grid:
            if signal and base in allowed:
                if gt == TARGET and base != TARGET:
                    route_stats[name]["gain_tp"] += 1
                elif gt != TARGET and base != TARGET:
                    route_stats[name]["add_fp"] += 1
                elif gt != TARGET and base == TARGET:
                    route_stats[name]["would_not_fix_existing_fp"] += 1
                elif gt == TARGET and base == TARGET:
                    route_stats[name]["already_tp"] += 1

        if gt == TARGET and base != TARGET:
            reason_missed_target[reason] += 1
            add_example(examples["missed_target"], (video, base, signal, reason, debug.get("channel"), best), args.max_examples)

        if reason == "no_pair_after_area_filter":
            no_pair_stats[gt].append(track_stats(lcrm))
            if gt == TARGET:
                add_example(examples["target_no_pair"], (video, base, track_stats(lcrm), debug), args.max_examples)
        if reason == "static_count_below_threshold":
            static_low_stats[gt].append(track_stats(lcrm))
            if gt == TARGET:
                add_example(examples["target_static_low"], (video, base, track_stats(lcrm), debug), args.max_examples)

    print("=== viability verdict inputs ===")
    print("signal_gt:", compact(signal_gt))
    print("signal_base:", compact(signal_base))
    print("signal_gt_base:", compact(signal_gt_base, 60))
    print("signal_precision_pool:", dict(signal_precision_pool))
    tp = signal_precision_pool["tp_signal"]
    fp = signal_precision_pool["fp_signal"]
    print("raw_signal_precision_if_all_flipped:", round(tp / (tp + fp), 4) if tp + fp else 0.0)
    print()

    print("=== route oracle stats ===")
    for name, stats in route_stats.items():
        gain = stats["gain_tp"]
        add = stats["add_fp"]
        print(name, dict(stats), "gain/add:", f"{gain}/{add}", "net:", gain - add)
    print()

    print("=== missed target reasons ===")
    print(compact(reason_missed_target))
    print()

    print("=== score margin quantiles (score - required_score) ===")
    for key, values in score_all.items():
        print(key, quantiles(values))
    print()

    print("=== best_pair field quantiles ===")
    for key, values in sorted(best_fields.items(), key=lambda x: str(x[0])):
        print(key, quantiles(values))
    print()

    print("=== near-threshold score_below candidates (-0.15 <= margin < 0) ===")
    for key, vals in sorted(score_near.items(), key=lambda x: str(x[0])):
        vals = sorted(vals, key=lambda x: x[0], reverse=True)
        print(str(key), "n=", len(vals))
        for margin, video, reason, best in vals[: args.max_examples]:
            print(" ", round(margin, 4), video, reason, best)
    print()

    print("=== no_pair track stats by gt ===")
    for gt, vals in no_pair_stats.items():
        print(gt, "n=", len(vals), {
            "static_count": quantiles([x["static_count"] for x in vals]),
            "static_tracks": quantiles([x["static_tracks"] for x in vals]),
            "track_count": quantiles([x["track_count"] for x in vals]),
            "frames_max": quantiles([x["frames_max"] for x in vals]),
            "step_min": quantiles([x["step_min"] for x in vals]),
        })
    print()

    print("=== static_low track stats by gt ===")
    for gt, vals in static_low_stats.items():
        print(gt, "n=", len(vals), {
            "static_count": quantiles([x["static_count"] for x in vals]),
            "static_tracks": quantiles([x["static_tracks"] for x in vals]),
            "track_count": quantiles([x["track_count"] for x in vals]),
            "frames_max": quantiles([x["frames_max"] for x in vals]),
            "step_min": quantiles([x["step_min"] for x in vals]),
        })
    print()

    for name in ["signal", "missed_target", "target_no_pair", "target_static_low"]:
        print(f"=== examples: {name} ===")
        for item in examples[name]:
            print(item)
        print()


if __name__ == "__main__":
    main()
