import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from eval_traffic_fixed import LABELS, UNKNOWN_LABEL, detect_label, get_gt_label

TARGET_LABEL = "多车事故"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def video_name(video: str) -> str:
    return Path(str(video).replace("\\", "/")).name


def event_ids(video: str) -> list[str]:
    return re.findall(r"\d{6,}", Path(video_name(video)).stem)


def load_review(path: Path) -> dict[str, dict[str, str]]:
    out = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row.get("event_id") or "").strip()
            if eid:
                out[eid] = {k: str(v or "").strip() for k, v in row.items()}
    return out


def is_strong_review(row: dict[str, str]) -> bool:
    return row.get("human_reason") == "collision_visible" or row.get("annotation_bucket") == "collision_process_clear"


def review_for_row(row: dict[str, Any], review: dict[str, dict[str, str]]) -> dict[str, str] | None:
    for eid in event_ids(str(row.get("video") or row.get("image") or "")):
        if eid in review:
            return review[eid]
    return None


def pct(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate multi-car accident recall by manual evidence group.")
    p.add_argument("--pred-jsonl", required=True)
    p.add_argument("--review-csv", required=True)
    p.add_argument("--pred-field", default="pred")
    p.add_argument("--gt-field", default="gt")
    p.add_argument("--output-json")
    p.add_argument("--show-examples", type=int, default=8)
    args = p.parse_args()

    rows = read_jsonl(Path(args.pred_jsonl))
    review = load_review(Path(args.review_csv))
    strong_ids = {eid for eid, row in review.items() if is_strong_review(row)}

    group_rows = {"全部多车事故": [], "弱证据多车事故": [], "强证据多车事故": []}
    fp_rows = []
    pred_multicar_total = 0
    pred_multicar_tp = 0

    for row in rows:
        gt = get_gt_label(row, args.gt_field) or UNKNOWN_LABEL
        pred = detect_label(row.get(args.pred_field)) or UNKNOWN_LABEL
        if pred == TARGET_LABEL:
            pred_multicar_total += 1
            if gt == TARGET_LABEL:
                pred_multicar_tp += 1
            else:
                fp_rows.append((row, gt, pred))
        if gt != TARGET_LABEL:
            continue

        rv = review_for_row(row, review)
        evidence_group = "强证据多车事故" if rv and is_strong_review(rv) else "弱证据多车事故"
        item = {
            "video": video_name(str(row.get("video") or row.get("image") or "")),
            "gt": gt,
            "pred": pred,
            "manual_reason": (rv or {}).get("human_reason", ""),
            "annotation_bucket": (rv or {}).get("annotation_bucket", ""),
            "pred_text": str(row.get(args.pred_field) or ""),
        }
        group_rows["全部多车事故"].append(item)
        group_rows[evidence_group].append(item)

    table = []
    for group, items in group_rows.items():
        support = len(items)
        hit = sum(1 for item in items if item["pred"] == TARGET_LABEL)
        table.append(
            {
                "group": group,
                "support": support,
                "pred_as_multicar": hit,
                "miss": support - hit,
                "recall": pct(hit, support),
                "pred_distribution": dict(Counter(item["pred"] for item in items)),
            }
        )

    precision = {
        "pred_multicar_total": pred_multicar_total,
        "true_multicar_among_pred": pred_multicar_tp,
        "false_positive": pred_multicar_total - pred_multicar_tp,
        "precision": pct(pred_multicar_tp, pred_multicar_total),
        "fp_by_gt": dict(Counter(gt for _, gt, _ in fp_rows)),
    }

    miss_examples = {}
    for group, items in group_rows.items():
        misses = [item for item in items if item["pred"] != TARGET_LABEL][: args.show_examples]
        miss_examples[group] = [
            {
                "video": item["video"],
                "pred": item["pred"],
                "manual_reason": item["manual_reason"],
                "annotation_bucket": item["annotation_bucket"],
                "pred_text": item["pred_text"][:300],
            }
            for item in misses
        ]

    result = {
        "input_file": args.pred_jsonl,
        "strong_collision_ids_total_in_review": len(strong_ids),
        "table": table,
        "multicar_prediction_precision": precision,
        "miss_examples": miss_examples,
    }

    print("=== multicar evidence group recall ===")
    print("group,support,pred_as_multicar,miss,recall,pred_distribution")
    for row in table:
        print(
            f"{row['group']},{row['support']},{row['pred_as_multicar']},"
            f"{row['miss']},{row['recall']},{row['pred_distribution']}"
        )
    print("\n=== multicar prediction precision ===")
    print(precision)
    print("\n=== miss examples ===")
    for group, examples in miss_examples.items():
        print("\n", group)
        for item in examples:
            print(item)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
