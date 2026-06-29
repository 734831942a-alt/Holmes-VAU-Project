import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_LABELS = [
    "正常",
    "拥堵",
    "异常停车",
    "占道施工",
    "交通事故",
    "二轮闯入",
    "行人闯入",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate traffic event detection metrics from batch_test_traffic prediction jsonl."
    )
    parser.add_argument("--pred-jsonl", required=True, help="Path to prediction jsonl")
    parser.add_argument(
        "--output-json",
        default="traffic_detection_eval.json",
        help="Where to save evaluation summary json",
    )
    parser.add_argument(
        "--error-csv",
        default="traffic_detection_errors.csv",
        help="Where to save misclassified sample list",
    )
    parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_LABELS),
        help="Comma-separated class labels",
    )
    parser.add_argument(
        "--normal-label",
        default="正常",
        help="The label considered as normal event",
    )
    parser.add_argument(
        "--pred-field",
        default="pred",
        help="Prediction text field in jsonl",
    )
    parser.add_argument(
        "--score-field",
        default="",
        help="Optional numeric score field for abnormal detection (higher=more abnormal)",
    )
    return parser.parse_args()


def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    return str(text).strip()


def extract_label_by_regex(text: str) -> Optional[str]:
    patterns = [
        r"事故类型\s*[：:]\s*[“\"']?([^，。,；;\"'”]+)",
        r"类型\s*[：:]\s*[“\"']?([^，。,；;\"'”]+)",
        r"类别\s*[：:]\s*[“\"']?([^，。,；;\"'”]+)",
        r"属于\s*[“\"']?([^，。,；;\"'”]+)",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1).strip()
    return None


def detect_label(text: str, labels: List[str]) -> Optional[str]:
    text = normalize_text(text)
    if not text:
        return None

    for lb in labels:
        if text == lb:
            return lb

    hit_labels = [lb for lb in labels if lb in text]
    if len(hit_labels) == 1:
        return hit_labels[0]

    candidate = extract_label_by_regex(text)
    if candidate:
        for lb in labels:
            if candidate == lb or lb in candidate:
                return lb

    if hit_labels:
        # Multiple labels appear, return first match to keep deterministic behavior.
        return hit_labels[0]
    return None


def get_gt_label(rec: Dict, labels: List[str]) -> Optional[str]:
    for k in ["gt_category", "label", "category", "gt_label"]:
        if k in rec:
            lb = detect_label(rec.get(k), labels)
            if lb:
                return lb
    for k in ["gt", "answer", "target"]:
        if k in rec:
            lb = detect_label(rec.get(k), labels)
            if lb:
                return lb
    return None


def get_pred_label(rec: Dict, labels: List[str], pred_field: str) -> Optional[str]:
    return detect_label(rec.get(pred_field), labels)


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_multiclass_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
) -> Dict:
    supports = {lb: 0 for lb in labels}
    tp = {lb: 0 for lb in labels}
    fp = {lb: 0 for lb in labels}
    fn = {lb: 0 for lb in labels}

    for t, p in zip(y_true, y_pred):
        supports[t] += 1
        if t == p:
            tp[t] += 1
        else:
            if p in fp:
                fp[p] += 1
            fn[t] += 1

    per_class = {}
    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
    weighted_p, weighted_r, weighted_f1 = 0.0, 0.0, 0.0
    total = len(y_true)

    for lb in labels:
        p = safe_div(tp[lb], tp[lb] + fp[lb])
        r = safe_div(tp[lb], tp[lb] + fn[lb])
        f1 = safe_div(2 * p * r, p + r)
        s = supports[lb]
        per_class[lb] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": s,
        }
        macro_p += p
        macro_r += r
        macro_f1 += f1
        weighted_p += p * s
        weighted_r += r * s
        weighted_f1 += f1 * s

    n_cls = len(labels) if labels else 1
    macro_p /= n_cls
    macro_r /= n_cls
    macro_f1 /= n_cls
    weighted_p = safe_div(weighted_p, total)
    weighted_r = safe_div(weighted_r, total)
    weighted_f1 = safe_div(weighted_f1, total)
    accuracy = safe_div(sum(int(t == p) for t, p in zip(y_true, y_pred)), total)

    matrix = {gt: {pd: 0 for pd in labels} for gt in labels}
    for t, p in zip(y_true, y_pred):
        if t in matrix and p in matrix[t]:
            matrix[t][p] += 1

    return {
        "accuracy": accuracy,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def compute_binary_abnormal_metrics(
    y_true: List[str],
    y_pred: List[str],
    normal_label: str,
) -> Dict:
    # positive class: abnormal
    tp = fp = fn = tn = 0
    y_true_bin = []
    y_pred_bin = []

    for t, p in zip(y_true, y_pred):
        tb = 0 if t == normal_label else 1
        pb = 0 if p == normal_label else 1
        y_true_bin.append(tb)
        y_pred_bin.append(pb)
        if tb == 1 and pb == 1:
            tp += 1
        elif tb == 0 and pb == 1:
            fp += 1
        elif tb == 1 and pb == 0:
            fn += 1
        else:
            tn += 1

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    acc = safe_div(tp + tn, len(y_true_bin))
    return {
        "positive_class": "abnormal",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def auc_roc(y_true_bin: List[int], scores: List[float]) -> Optional[float]:
    pos = sum(y_true_bin)
    neg = len(y_true_bin) - pos
    if pos == 0 or neg == 0:
        return None
    pairs = sorted(zip(scores, y_true_bin), key=lambda x: x[0])
    rank_sum = 0.0
    for idx, (_, y) in enumerate(pairs, start=1):
        if y == 1:
            rank_sum += idx
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def auc_pr(y_true_bin: List[int], scores: List[float]) -> Optional[float]:
    pos = sum(y_true_bin)
    if pos == 0:
        return None
    pairs = sorted(zip(scores, y_true_bin), key=lambda x: x[0], reverse=True)
    tp = fp = 0
    prev_recall = 0.0
    area = 0.0
    for _, y in pairs:
        if y == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / pos
        precision = tp / (tp + fp)
        area += (recall - prev_recall) * precision
        prev_recall = recall
    return area


def main():
    args = parse_args()
    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    if args.normal_label not in labels:
        raise ValueError(f"--normal-label={args.normal_label} 不在 labels 中")

    pred_path = Path(args.pred_jsonl)
    rows = []
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    total = len(rows)
    with_gt = 0
    y_true = []
    y_pred = []
    unknown_pred = 0
    unknown_gt = 0
    errors = []

    score_vals: List[float] = []
    score_gt_bin: List[int] = []
    use_score = bool(args.score_field)

    for i, rec in enumerate(rows):
        gt = get_gt_label(rec, labels)
        pred = get_pred_label(rec, labels, args.pred_field)

        if gt is None:
            unknown_gt += 1
            continue
        with_gt += 1

        if pred is None:
            unknown_pred += 1
            pred = "__UNK__"

        y_true.append(gt)
        y_pred.append(pred)

        if gt != pred:
            errors.append(
                {
                    "index": rec.get("index", i),
                    "id": rec.get("id", ""),
                    "video": rec.get("video", ""),
                    "gt_label": gt,
                    "pred_label": pred,
                    "pred_text": normalize_text(rec.get(args.pred_field, "")),
                }
            )

        if use_score and args.score_field in rec:
            v = rec.get(args.score_field)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(float(v)):
                score_vals.append(float(v))
                score_gt_bin.append(0 if gt == args.normal_label else 1)

    # Keep only known labels for multiclass metrics. Unknown predictions are counted as wrong in accuracy
    # but ignored in per-class confusion (since __UNK__ not in labels).
    y_pred_known = [p if p in labels else args.normal_label for p in y_pred]
    mc = compute_multiclass_metrics(y_true, y_pred_known, labels)
    binary = compute_binary_abnormal_metrics(y_true, y_pred_known, args.normal_label)

    score_metrics = {}
    if len(score_vals) == len(score_gt_bin) and len(score_vals) > 0:
        score_metrics["auroc_abnormal"] = auc_roc(score_gt_bin, score_vals)
        score_metrics["auprc_abnormal"] = auc_pr(score_gt_bin, score_vals)
        score_metrics["score_field"] = args.score_field

    summary = {
        "input_file": str(pred_path),
        "labels": labels,
        "normal_label": args.normal_label,
        "total_records": total,
        "records_with_gt_label": with_gt,
        "unknown_gt_count": unknown_gt,
        "unknown_pred_count": unknown_pred,
        "multiclass_detection": mc,
        "binary_abnormal_detection": binary,
        "score_based_metrics": score_metrics,
        "num_errors": len(errors),
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    err_csv = Path(args.error_csv)
    err_csv.parent.mkdir(parents=True, exist_ok=True)
    with err_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "id", "video", "gt_label", "pred_label", "pred_text"],
        )
        writer.writeheader()
        for e in errors:
            writer.writerow(e)

    print("Done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved summary to: {out_json}")
    print(f"Saved errors to: {err_csv}")


if __name__ == "__main__":
    main()
