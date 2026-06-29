import argparse
import json
from collections import Counter
from pathlib import Path

from eval_traffic_fixed import LABELS, UNKNOWN_LABEL, detect_label, get_gt_label
from postprocess_lcrm_lite_fusion import get_ovd_payload, lcrm_lite_signal, parse_csv, read_jsonl


def by_video(rows):
    return {row.get("video"): row for row in rows if row.get("video")}


def label_or_unknown(text):
    return detect_label(text) or UNKNOWN_LABEL


def pct(num, den):
    return round(100.0 * num / den, 2) if den else 0.0


def compact_counter(counter, limit=30):
    return {str(k): v for k, v in counter.most_common(limit)}


def main():
    ap = argparse.ArgumentParser(description="Diagnose LCRM coverage before final fusion routing.")
    ap.add_argument("--base-pred", required=True)
    ap.add_argument("--ovd", required=True)
    ap.add_argument("--base-pred-field", default="pred")
    ap.add_argument("--gt-field", default="gt")
    ap.add_argument("--allowed-base-labels", default="异常停车", type=parse_csv)
    ap.add_argument("--source", default="lcrm")
    ap.add_argument("--max-examples", type=int, default=25)

    # Keep this in sync with postprocess_lcrm_lite_fusion.py options used by lcrm_lite_signal.
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

    total = 0
    matched_ovd = 0
    lcrm_present = 0
    lcrm_ok = 0
    signal_total = 0
    applicable_total = 0
    missed_target = 0
    missed_target_signal = 0
    missed_target_applicable = 0

    gt_counts = Counter()
    base_counts = Counter()
    reason_counts = Counter()
    reason_by_gt = Counter()
    reason_by_base = Counter()
    signal_by_gt = Counter()
    signal_by_base = Counter()
    applicable_by_gt = Counter()
    applicable_by_base = Counter()
    missed_reason = Counter()
    missed_base = Counter()
    debug_channel = Counter()

    examples_signal_not_applicable = []
    examples_applicable = []
    examples_missed_target = []

    for row in rows:
        total += 1
        video = row.get("video", "")
        gt = get_gt_label(row, args.gt_field)
        base_label = label_or_unknown(row.get(args.base_pred_field))
        gt_counts[gt] += 1
        base_counts[base_label] += 1

        ovd = get_ovd_payload(ovd_map.get(video, {}))
        if ovd:
            matched_ovd += 1
        lcrm = ovd.get("lcrm") if isinstance(ovd, dict) else None
        if isinstance(lcrm, dict):
            lcrm_present += 1
            if lcrm.get("ok"):
                lcrm_ok += 1

        signal, debug = lcrm_lite_signal(lcrm, args)
        reason = debug.get("reason")
        channel = debug.get("channel")
        reason_counts[reason] += 1
        reason_by_gt[(gt, reason)] += 1
        reason_by_base[(base_label, reason)] += 1
        debug_channel[channel] += 1

        if signal:
            signal_total += 1
            signal_by_gt[gt] += 1
            signal_by_base[base_label] += 1

        applicable = signal and base_label in args.allowed_base_labels
        if applicable:
            applicable_total += 1
            applicable_by_gt[gt] += 1
            applicable_by_base[base_label] += 1
            if len(examples_applicable) < args.max_examples:
                examples_applicable.append((video, gt, base_label, reason, channel, debug.get("best_pair")))
        elif signal and len(examples_signal_not_applicable) < args.max_examples:
            examples_signal_not_applicable.append((video, gt, base_label, reason, channel, debug.get("best_pair")))

        if gt == "多车事故" and base_label != "多车事故":
            missed_target += 1
            missed_reason[reason] += 1
            missed_base[base_label] += 1
            if signal:
                missed_target_signal += 1
            if applicable:
                missed_target_applicable += 1
            if len(examples_missed_target) < args.max_examples:
                examples_missed_target.append((video, base_label, signal, applicable, reason, channel, debug.get("best_pair")))

    print("=== coverage summary ===")
    print("total:", total)
    print("ovd_matched:", matched_ovd, f"({pct(matched_ovd,total)}%)")
    print("lcrm_present:", lcrm_present, f"({pct(lcrm_present,total)}%)")
    print("lcrm_ok:", lcrm_ok, f"({pct(lcrm_ok,total)}%)")
    print("signal_total:", signal_total, f"({pct(signal_total,total)}%)")
    print("applicable_total:", applicable_total, f"({pct(applicable_total,total)}%)")
    print("allowed_base_labels:", args.allowed_base_labels)
    print()
    print("gt_counts:", compact_counter(gt_counts))
    print("base_counts:", compact_counter(base_counts))
    print("reason_counts:", compact_counter(reason_counts))
    print("channel_counts:", compact_counter(debug_channel))
    print()
    print("signal_by_gt:", compact_counter(signal_by_gt))
    print("signal_by_base:", compact_counter(signal_by_base))
    print("applicable_by_gt:", compact_counter(applicable_by_gt))
    print("applicable_by_base:", compact_counter(applicable_by_base))
    print()
    print("=== missed true multicar coverage ===")
    print("missed_target_base_not_multicar:", missed_target)
    print("missed_target_signal:", missed_target_signal, f"({pct(missed_target_signal, missed_target)}%)")
    print("missed_target_applicable:", missed_target_applicable, f"({pct(missed_target_applicable, missed_target)}%)")
    print("missed_base:", compact_counter(missed_base))
    print("missed_reason:", compact_counter(missed_reason))
    print()
    print("reason_by_gt:", compact_counter(reason_by_gt, 50))
    print("reason_by_base:", compact_counter(reason_by_base, 50))
    print()
    print("=== applicable examples ===")
    for x in examples_applicable:
        print(x)
    print()
    print("=== signal but not applicable examples ===")
    for x in examples_signal_not_applicable:
        print(x)
    print()
    print("=== missed true multicar examples ===")
    for x in examples_missed_target:
        print(x)


if __name__ == "__main__":
    main()
