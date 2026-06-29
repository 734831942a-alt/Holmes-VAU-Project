import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from eval_traffic_fixed import LABELS, get_gt_label


MULTICAR_LABEL = LABELS[0]
DATASET_PROTOCOL = "traffic_e9_multicar_clean_v1"


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_text(path, values):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for value in values:
            f.write(str(value) + "\n")


def event_ids_from_video(video):
    stem = Path(str(video or "")).stem
    return re.findall(r"\d{6,}", stem)


def main_event_id(row):
    ids = event_ids_from_video(row.get("video") or row.get("video_path") or row.get("image") or "")
    return ids[-1] if ids else ""


def read_review_csv(path):
    review = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            event_id = str(row.get("event_id") or "").strip()
            if not event_id:
                continue
            keep = str(row.get("human_keep") or "").strip().lower()
            reason = str(row.get("human_reason") or "").strip()
            if not keep:
                keep = str(row.get("auto_suggest") or "").strip().lower()
                if keep == "keep":
                    keep = "yes"
                elif keep == "drop":
                    keep = "no"
            review[event_id] = {
                "event_id": event_id,
                "human_keep": keep,
                "human_reason": reason,
                "manual_note": str(row.get("manual_note") or "").strip(),
                "annotation_bucket": str(row.get("annotation_bucket") or "").strip(),
                "auto_suggest": str(row.get("auto_suggest") or "").strip(),
                "auto_reason": str(row.get("auto_reason") or "").strip(),
            }
    return review


def attach_review(row, info):
    out = dict(row)
    out["multicar_review_event_id"] = info.get("event_id", "")
    out["multicar_review_keep"] = info.get("human_keep", "")
    out["multicar_review_reason"] = info.get("human_reason", "")
    out["multicar_review_bucket"] = info.get("annotation_bucket", "")
    out["multicar_review_note"] = info.get("manual_note", "")
    return out


def split_rows(rows, review, args):
    clean = []
    removed = []
    uncertain = []
    unreviewed = []
    stats = Counter()
    reason_counts = Counter()
    id_to_action = {}

    for row in rows:
        gt = get_gt_label(row, args.gt_field)
        stats["total"] += 1
        stats[f"gt::{gt or '__UNKNOWN__'}"] += 1

        if gt != MULTICAR_LABEL:
            clean.append(row)
            stats["non_multicar_kept"] += 1
            continue

        stats["multicar_total"] += 1
        ids = event_ids_from_video(row.get("video") or row.get("video_path") or row.get("image") or "")
        info = next((review[x] for x in reversed(ids) if x in review), None)

        if not info:
            stats["multicar_unreviewed"] += 1
            action = args.unreviewed_policy
            rec = dict(row)
            rec["multicar_review_event_id"] = ids[-1] if ids else ""
            rec["multicar_review_keep"] = "unreviewed"
            rec["multicar_review_reason"] = "unreviewed"
            if action == "keep":
                clean.append(rec)
                stats["multicar_unreviewed_kept"] += 1
            elif action == "drop":
                removed.append(rec)
                stats["multicar_unreviewed_removed"] += 1
            else:
                unreviewed.append(rec)
                stats["multicar_unreviewed_separate"] += 1
            continue

        keep = info.get("human_keep", "")
        reason = info.get("human_reason", "")
        reason_counts[(keep, reason)] += 1
        rec = attach_review(row, info)
        event_id = info.get("event_id", "")

        if keep == "yes":
            clean.append(rec)
            stats["multicar_yes_kept"] += 1
            id_to_action[event_id] = "keep"
        elif keep == "no":
            removed.append(rec)
            stats["multicar_no_removed"] += 1
            id_to_action[event_id] = "drop"
        elif keep == "uncertain":
            if args.uncertain_policy == "keep":
                clean.append(rec)
                stats["multicar_uncertain_kept"] += 1
                id_to_action[event_id] = "keep_uncertain"
            elif args.uncertain_policy == "drop":
                removed.append(rec)
                stats["multicar_uncertain_removed"] += 1
                id_to_action[event_id] = "drop_uncertain"
            else:
                uncertain.append(rec)
                stats["multicar_uncertain_separate"] += 1
                id_to_action[event_id] = "uncertain"
        else:
            unreviewed.append(rec)
            stats["multicar_bad_review_value"] += 1
            id_to_action[event_id] = "bad_review_value"

    return clean, removed, uncertain, unreviewed, stats, reason_counts, id_to_action


def gt_distribution(rows, gt_field):
    counts = Counter()
    for row in rows:
        counts[get_gt_label(row, gt_field) or "__UNKNOWN__"] += 1
    return dict(counts)


def output_stem(path, suffix):
    stem = Path(path).stem
    return f"{stem}_{suffix}" if suffix else stem


def process_one(input_jsonl, review, args):
    rows = read_jsonl(input_jsonl)
    clean, removed, uncertain, unreviewed, stats, reason_counts, id_to_action = split_rows(rows, review, args)

    stem = output_stem(input_jsonl, args.suffix)
    out_dir = Path(args.output_dir)
    paths = {
        "clean": out_dir / f"{stem}.jsonl",
        "removed": out_dir / f"{stem}_removed_multicar.jsonl",
        "uncertain": out_dir / f"{stem}_uncertain_multicar.jsonl",
        "unreviewed": out_dir / f"{stem}_unreviewed_multicar.jsonl",
        "summary": out_dir / f"{stem}_summary.json",
    }

    write_jsonl(paths["clean"], clean)
    write_jsonl(paths["removed"], removed)
    write_jsonl(paths["uncertain"], uncertain)
    write_jsonl(paths["unreviewed"], unreviewed)

    summary = {
        "dataset_protocol": DATASET_PROTOCOL,
        "input_jsonl": str(input_jsonl),
        "output_clean": str(paths["clean"]),
        "output_removed": str(paths["removed"]),
        "output_uncertain": str(paths["uncertain"]),
        "output_unreviewed": str(paths["unreviewed"]),
        "uncertain_policy": args.uncertain_policy,
        "unreviewed_policy": args.unreviewed_policy,
        "stats": dict(stats),
        "review_reason_counts": {str(k): v for k, v in reason_counts.most_common()},
        "counts": {
            "input": len(rows),
            "clean": len(clean),
            "removed": len(removed),
            "uncertain": len(uncertain),
            "unreviewed": len(unreviewed),
        },
        "label_distribution": {
            "input": gt_distribution(rows, args.gt_field),
            "clean": gt_distribution(clean, args.gt_field),
            "removed": gt_distribution(removed, args.gt_field),
            "uncertain": gt_distribution(uncertain, args.gt_field),
            "unreviewed": gt_distribution(unreviewed, args.gt_field),
        },
    }
    paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    write_text(out_dir / f"{stem}_keep_ids.txt", sorted(k for k, v in id_to_action.items() if v.startswith("keep")))
    write_text(out_dir / f"{stem}_drop_ids.txt", sorted(k for k, v in id_to_action.items() if v.startswith("drop")))
    write_text(out_dir / f"{stem}_uncertain_ids.txt", sorted(k for k, v in id_to_action.items() if v == "uncertain"))

    print("=" * 100)
    print("input:", input_jsonl)
    print("clean:", paths["clean"], "n=", len(clean))
    print("removed:", paths["removed"], "n=", len(removed))
    print("uncertain:", paths["uncertain"], "n=", len(uncertain))
    print("unreviewed:", paths["unreviewed"], "n=", len(unreviewed))
    print("summary:", paths["summary"])
    print("stats:", json.dumps(dict(stats), ensure_ascii=False))
    return summary


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Rebuild traffic JSONL datasets using human-reviewed multicar evidence labels. "
            "Default clean set keeps reviewed yes samples and separates no/uncertain multicar samples."
        )
    )
    ap.add_argument("--review-csv", required=True)
    ap.add_argument("--input-jsonl", required=True, action="append", help="Can be passed multiple times.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--gt-field", default="gt")
    ap.add_argument("--suffix", default="e9_multicar_clean_v1")
    ap.add_argument("--uncertain-policy", choices=["separate", "keep", "drop"], default="separate")
    ap.add_argument("--unreviewed-policy", choices=["separate", "keep", "drop"], default="keep")
    args = ap.parse_args()

    review = read_review_csv(args.review_csv)
    print("review_csv:", args.review_csv, "n=", len(review))
    print("positive_review:", sum(1 for x in review.values() if x.get("human_keep") == "yes"))
    print("negative_review:", sum(1 for x in review.values() if x.get("human_keep") == "no"))
    print("uncertain_review:", sum(1 for x in review.values() if x.get("human_keep") == "uncertain"))

    all_summaries = []
    for input_jsonl in args.input_jsonl:
        all_summaries.append(process_one(input_jsonl, review, args))

    used_review_ids = set()
    for summary in all_summaries:
        for key in ("output_clean", "output_removed", "output_uncertain", "output_unreviewed"):
            path = summary.get(key)
            if not path or not Path(path).exists():
                continue
            for row in read_jsonl(path):
                event_id = str(row.get("multicar_review_event_id") or "").strip()
                if event_id in review:
                    used_review_ids.add(event_id)

    unused_review_ids = sorted(set(review) - used_review_ids)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = output_dir / "rebuild_multicar_review_summary.json"
    combined_payload = {
        "dataset_protocol": DATASET_PROTOCOL,
        "review_csv": args.review_csv,
        "review_count": len(review),
        "used_review_count": len(used_review_ids),
        "unused_review_count": len(unused_review_ids),
        "unused_review_ids": unused_review_ids,
        "summaries": all_summaries,
    }
    combined.write_text(json.dumps(combined_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("combined_summary:", combined)
    print("review_used:", len(used_review_ids), "/", len(review), "unused:", len(unused_review_ids))


if __name__ == "__main__":
    main()
