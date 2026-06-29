import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from eval_traffic_fixed import LABELS, UNKNOWN_LABEL, detect_label, get_gt_label


TARGET = "多车事故"


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def by_video(rows):
    return {row.get("video"): row for row in rows if row.get("video")}


def label_or_unknown(text):
    return detect_label(text) or UNKNOWN_LABEL


def metric_for_target(pairs):
    tp = fp = fn = 0
    for gt, pred in pairs:
        if gt == TARGET and pred == TARGET:
            tp += 1
        elif gt != TARGET and pred == TARGET:
            fp += 1
        elif gt == TARGET and pred != TARGET:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall, "f1": f1}


def short(text, n=220):
    text = "" if text is None else str(text).replace("\n", " ")
    return text[:n]


def main():
    ap = argparse.ArgumentParser(description="Diagnose LCRM effect under eval_traffic_fixed.py label rules.")
    ap.add_argument("--base", required=True, help="Baseline prediction jsonl, e.g. traffic_test_pred_e8.jsonl")
    ap.add_argument("--lcrm", required=True, help="LCRM fused jsonl")
    ap.add_argument("--base-field", default="pred")
    ap.add_argument("--lcrm-field", default="pred_lcrm_lite_fused")
    ap.add_argument("--gt-field", default="gt")
    ap.add_argument("--max-examples", type=int, default=30)
    args = ap.parse_args()

    base_rows = read_jsonl(args.base)
    lcrm_map = by_video(read_jsonl(args.lcrm))

    base_pairs = []
    lcrm_pairs = []
    changed = []
    missing_lcrm = 0

    for idx, base in enumerate(base_rows):
        video = base.get("video")
        lcrm = lcrm_map.get(video)
        if lcrm is None:
            missing_lcrm += 1
            continue
        gt = get_gt_label(base, args.gt_field)
        if gt not in LABELS:
            continue
        base_pred = label_or_unknown(base.get(args.base_field))
        lcrm_pred = label_or_unknown(lcrm.get(args.lcrm_field))
        base_pairs.append((gt, base_pred))
        lcrm_pairs.append((gt, lcrm_pred))
        if base_pred != lcrm_pred:
            debug = lcrm.get("lcrm_lite_debug") or {}
            changed.append(
                {
                    "video": video,
                    "gt": gt,
                    "base_pred": base_pred,
                    "lcrm_pred": lcrm_pred,
                    "reason": debug.get("reason"),
                    "signal": lcrm.get("lcrm_lite_signal"),
                    "base_label": lcrm.get("lcrm_lite_base_label"),
                    "fused_label": lcrm.get("lcrm_lite_fused_label"),
                    "channel": debug.get("channel"),
                    "best_pair": debug.get("best_pair"),
                    "base_text": short(base.get(args.base_field)),
                    "lcrm_text": short(lcrm.get(args.lcrm_field)),
                }
            )

    base_metric = metric_for_target(base_pairs)
    lcrm_metric = metric_for_target(lcrm_pairs)

    categories = Counter()
    reason = Counter()
    gt_base_lcrm = Counter()
    for item in changed:
        gt = item["gt"]
        b = item["base_pred"]
        f = item["lcrm_pred"]
        if gt == TARGET and b != TARGET and f == TARGET:
            cat = "gain_tp"
        elif gt == TARGET and b == TARGET and f != TARGET:
            cat = "lose_tp"
        elif gt != TARGET and b != TARGET and f == TARGET:
            cat = "add_fp"
        elif gt != TARGET and b == TARGET and f != TARGET:
            cat = "remove_fp"
        else:
            cat = "other_change"
        categories[cat] += 1
        reason[item["reason"]] += 1
        gt_base_lcrm[(gt, b, f)] += 1

    missed_multicar = []
    for idx, base in enumerate(base_rows):
        video = base.get("video")
        lcrm = lcrm_map.get(video)
        if not lcrm:
            continue
        gt = get_gt_label(base, args.gt_field)
        if gt != TARGET:
            continue
        base_pred = label_or_unknown(base.get(args.base_field))
        lcrm_pred = label_or_unknown(lcrm.get(args.lcrm_field))
        if base_pred != TARGET and lcrm_pred != TARGET:
            debug = lcrm.get("lcrm_lite_debug") or {}
            missed_multicar.append(
                {
                    "video": video,
                    "base_pred": base_pred,
                    "lcrm_pred": lcrm_pred,
                    "reason": debug.get("reason"),
                    "signal": lcrm.get("lcrm_lite_signal"),
                    "base_label": lcrm.get("lcrm_lite_base_label"),
                    "channel": debug.get("channel"),
                    "best_pair": debug.get("best_pair"),
                    "debug": debug,
                    "text": short(lcrm.get(args.lcrm_field)),
                }
            )

    print("=== fixed-label target metric ===")
    print("base:", base_metric)
    print("lcrm:", lcrm_metric)
    print("delta:", {k: round(lcrm_metric[k] - base_metric[k], 6) for k in base_metric})
    print()
    print("matched:", len(base_pairs), "missing_lcrm:", missing_lcrm)
    print("changed_count:", len(changed))
    print("change_categories:", dict(categories))
    print("changed_reason:", dict(reason))
    print("changed_gt_base_lcrm:", {str(k): v for k, v in gt_base_lcrm.items()})
    print()
    print("=== changed examples ===")
    for item in changed[: args.max_examples]:
        print("=" * 100)
        print("video:", item["video"])
        print("gt/base/lcrm:", item["gt"], item["base_pred"], "->", item["lcrm_pred"])
        print("reason/signal/channel:", item["reason"], item["signal"], item["channel"])
        print("base_label/fused_label:", item["base_label"], item["fused_label"])
        print("best_pair:", item["best_pair"])
        print("base_text:", item["base_text"])
        print("lcrm_text:", item["lcrm_text"])
    print()
    print("=== missed true multicar still not fixed ===")
    print("count:", len(missed_multicar))
    print("reason_counts:", dict(Counter(x["reason"] for x in missed_multicar)))
    print("base_pred_counts:", dict(Counter(x["base_pred"] for x in missed_multicar)))
    for item in missed_multicar[: args.max_examples]:
        print("=" * 100)
        print("video:", item["video"])
        print("base/lcrm:", item["base_pred"], "->", item["lcrm_pred"])
        print("reason/signal/channel/base_label:", item["reason"], item["signal"], item["channel"], item["base_label"])
        print("best_pair:", item["best_pair"])
        print("text:", item["text"])


if __name__ == "__main__":
    main()
