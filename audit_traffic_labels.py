import argparse
import json
import re
from collections import Counter
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Audit label distribution from traffic jsonl files.")
    p.add_argument("--jsonl", required=True, help="Path to jsonl")
    p.add_argument("--topk", type=int, default=30, help="Top-K to print")
    return p.parse_args()


def file_prefix_label(video_name: str) -> str:
    if not video_name:
        return ""
    base = Path(video_name).name
    if "_" in base:
        return base.split("_", 1)[0].strip()
    return base.strip()


def extract_label_from_text(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"(事故类型|类型|类别)\s*[：:]\s*[“\"']?([^，。,；;\"'”]+)", text)
    if m:
        return m.group(2).strip()
    return ""


def main():
    args = parse_args()
    p = Path(args.jsonl)
    rows = []
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass

    c_video_prefix = Counter()
    c_gt_category_field = Counter()
    c_gt_text_extract = Counter()

    gt_missing = 0
    parse_fail = 0
    for r in rows:
        video = str(r.get("video", ""))
        if video:
            c_video_prefix[file_prefix_label(video)] += 1

        gt_cat = str(r.get("gt_category", "")).strip()
        if gt_cat:
            c_gt_category_field[gt_cat] += 1
        else:
            gt_missing += 1

        gt = str(r.get("gt", ""))
        ex = extract_label_from_text(gt)
        if ex:
            c_gt_text_extract[ex] += 1
        else:
            parse_fail += 1

    print(f"file: {p}")
    print(f"records: {len(rows)}")
    print(f"gt_category missing: {gt_missing}")
    print(f"gt text label parse fail: {parse_fail}")

    print("\n[video filename prefix distribution]")
    for k, v in c_video_prefix.most_common(args.topk):
        print(f"{v}\t{k}")

    print("\n[gt_category field distribution]")
    for k, v in c_gt_category_field.most_common(args.topk):
        print(f"{v}\t{k}")

    print("\n[label extracted from gt text]")
    for k, v in c_gt_text_extract.most_common(args.topk):
        print(f"{v}\t{k}")


if __name__ == "__main__":
    main()

