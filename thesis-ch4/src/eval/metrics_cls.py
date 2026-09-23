# Frozen transplant from eval_traffic_fixed.py
# Source commit: 8b98e6cda09d078061eb308d78feec75fa23e6fd
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PROTOCOL_NAME = "traffic_abnormal_fixed_v1"

from src.evidence.queries import ALIASES, LABELS
UNKNOWN_LABEL = "__UNKNOWN__"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Fixed-protocol traffic abnormal evaluation. "
            "Unknown predictions are counted as wrong but are not remapped to a real label."
        )
    )
    ap.add_argument("--pred-jsonl", required=True)
    ap.add_argument("--pred-field", default="pred")
    ap.add_argument("--gt-field", default="gt")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--error-csv", required=True)
    return ap.parse_args()


def norm(value: object) -> str:
    return "" if value is None else str(value).strip()


def filename_prefix(video: object) -> Optional[str]:
    name = Path(norm(video)).name
    if "_" not in name:
        return None
    prefix = name.split("_", 1)[0]
    return prefix if prefix in LABELS else None


def key_to_label() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for label in LABELS:
        mapping[label] = label
        for alias in ALIASES.get(label, []):
            mapping[alias] = label
    return mapping


def detect_in_span(span: str, mapping: Dict[str, str]) -> Optional[str]:
    span = norm(span)
    if not span:
        return None
    if span in mapping:
        return mapping[span]
    for key in sorted(mapping, key=len, reverse=True):
        if key in span:
            return mapping[key]
    return None


def detect_last_mention(text: str, mapping: Dict[str, str]) -> Optional[str]:
    best: Tuple[int, int, Optional[str]] = (-1, -1, None)
    for key, label in mapping.items():
        pos = text.rfind(key)
        if pos >= 0:
            # Later mention wins; if tied, longer key wins.
            cand = (pos, len(key), label)
            if cand[:2] > best[:2]:
                best = cand
    return best[2]


def detect_label(text: object) -> Optional[str]:
    text = norm(text)
    if not text:
        return None
    mapping = key_to_label()

    if text in mapping:
        return mapping[text]

    # Structured model outputs often put the class on item 2.
    for line in text.splitlines():
        m = re.match(r"^\s*2\s*[\.\、:：\)]\s*(.+?)\s*$", line)
        if m:
            label = detect_in_span(m.group(1), mapping)
            if label:
                return label

    strong_patterns = [
        r"(?:事件类型|事故类型|类别|类型)\s*[:：]\s*([^。；;\n\r]+)",
        r"(?:判定为|判断为|识别为|属于|本次事件为|该事件为)\s*([^。；;\n\r]+)",
    ]
    hits: List[Tuple[int, str]] = []
    for pattern in strong_patterns:
        for match in re.finditer(pattern, text):
            label = detect_in_span(match.group(1), mapping)
            if label:
                hits.append((match.start(), label))
    if hits:
        return sorted(hits, key=lambda x: x[0])[-1][1]

    return detect_last_mention(text, mapping)


def get_gt_label(row: Dict, gt_field: str) -> Optional[str]:
    label = filename_prefix(row.get("video") or row.get("video_path"))
    if label:
        return label
    for key in ("gt_label", "gt_category", "label", "category"):
        if key in row:
            label = detect_label(row.get(key))
            if label:
                return label
    return detect_label(row.get(gt_field))


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    support = {label: 0 for label in LABELS}
    tp = {label: 0 for label in LABELS}
    fp = {label: 0 for label in LABELS}
    fn = {label: 0 for label in LABELS}
    confusion = {label: {pred: 0 for pred in LABELS + [UNKNOWN_LABEL]} for label in LABELS}

    for gt, pred in zip(y_true, y_pred):
        support[gt] += 1
        confusion[gt][pred] += 1
        if pred == gt:
            tp[gt] += 1
        else:
            fn[gt] += 1
            if pred in fp:
                fp[pred] += 1

    per_class = {}
    macro_f1 = 0.0
    for label in LABELS:
        precision = safe_div(tp[label], tp[label] + fp[label])
        recall = safe_div(tp[label], tp[label] + fn[label])
        f1 = safe_div(2 * precision * recall, precision + recall)
        macro_f1 += f1
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support[label],
        }

    total = len(y_true)
    return {
        "accuracy": safe_div(sum(1 for gt, pred in zip(y_true, y_pred) if gt == pred), total),
        "macro_f1": safe_div(macro_f1, len(LABELS)),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def read_jsonl(path: str) -> Iterable[Dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    args = parse_args()
    rows = list(read_jsonl(args.pred_jsonl))

    y_true: List[str] = []
    y_pred: List[str] = []
    unknown_gt = 0
    unknown_pred = 0
    errors: List[Dict[str, object]] = []

    for idx, row in enumerate(rows):
        gt = get_gt_label(row, args.gt_field)
        pred = detect_label(row.get(args.pred_field))
        if gt is None:
            unknown_gt += 1
            continue
        if pred is None:
            unknown_pred += 1
            pred = UNKNOWN_LABEL

        y_true.append(gt)
        y_pred.append(pred)

        if gt != pred:
            errors.append(
                {
                    "index": row.get("index", idx),
                    "id": row.get("id", ""),
                    "video": row.get("video", ""),
                    "gt_label": gt,
                    "pred_label": pred,
                    "pred_text": norm(row.get(args.pred_field)),
                }
            )

    summary = {
        "eval_protocol": PROTOCOL_NAME,
        "unknown_prediction_policy": "count_as_wrong_without_mapping_to_real_label",
        "input_file": args.pred_jsonl,
        "pred_field": args.pred_field,
        "gt_field": args.gt_field,
        "labels": LABELS,
        "label_aliases": ALIASES,
        "total_records": len(rows),
        "valid_records_for_classification": len(y_true),
        "unknown_gt_count": unknown_gt,
        "unknown_pred_count": unknown_pred,
        "multiclass_detection": compute_metrics(y_true, y_pred) if y_true else {},
        "num_errors": len(errors),
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    err_csv = Path(args.error_csv)
    err_csv.parent.mkdir(parents=True, exist_ok=True)
    with err_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["index", "id", "video", "gt_label", "pred_label", "pred_text"]
        )
        writer.writeheader()
        writer.writerows(errors)

    print("Done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved summary: {out_json}")
    print(f"Saved errors : {err_csv}")


if __name__ == "__main__":
    main()
