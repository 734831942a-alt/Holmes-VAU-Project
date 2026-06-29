import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_LABELS = [
    "多车事故",
    "拥堵",
    "异常停车",
    "占道施工",
    "二轮车辆闯入",
    "抛洒物",
]


LABEL_ALIASES = {
    "多车事故": ["多车事故", "交通事故", "事故", "碰撞", "追尾", "accident"],
    "拥堵": ["拥堵", "堵车", "缓行", "congestion", "traffic jam"],
    "异常停车": ["异常停车", "停车", "静止", "parking", "stopped vehicle"],
    "占道施工": ["占道施工", "施工", "锥桶", "construction"],
    "二轮车辆闯入": ["二轮车辆闯入", "二轮", "摩托车", "电动车", "motorcycle"],
    "抛洒物": ["抛洒物", "抛撒物", "散落物", "异物", "debris", "obstacle"],
}


def parse_csv(text):
    return [x.strip() for x in str(text or "").split(",") if x.strip()]


def parse_label_counts(text):
    counts = {}
    for item in str(text or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --per-label-counts item: {item!r}; expected label=count")
        label, value = item.split("=", 1)
        counts[label.strip()] = int(value.strip())
    return counts


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


def conversation_value(row, role):
    for item in row.get("conversations", []) or []:
        if item.get("from") == role:
            return str(item.get("value") or "")
    return ""


def detect_label(row, labels):
    video = str(row.get("video") or row.get("image") or "")
    name = Path(video).name
    if "_" in name:
        prefix = name.split("_", 1)[0]
        if prefix in labels:
            return prefix

    text = " ".join([
        str(row.get("gt") or ""),
        str(row.get("label") or ""),
        conversation_value(row, "gpt"),
    ])
    for label in labels:
        if label in text:
            return label
    lowered = text.lower()
    hits = []
    for label in labels:
        for alias in LABEL_ALIASES.get(label, [label]):
            if alias.lower() in lowered:
                hits.append(label)
                break
    if len(hits) == 1:
        return hits[0]
    return None


def normalize_row(row, idx):
    out = dict(row)
    video = out.get("video") or out.get("image") or ""
    out["video"] = video
    out.setdefault("id", idx)
    if not out.get("prompt"):
        prompt = conversation_value(out, "human")
        prompt = re.sub(r"<\s*(image|video)\s*>", "", prompt).strip()
        if prompt:
            out["prompt"] = prompt
    if not out.get("gt"):
        gt = conversation_value(out, "gpt")
        if gt:
            out["gt"] = gt
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Create a label-balanced validation split for LCRM threshold tuning."
    )
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--val-out", required=True)
    ap.add_argument("--train-out", default="")
    ap.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--per-label", type=int, default=0, help="Fixed validation count per label; 0 uses val-ratio.")
    ap.add_argument(
        "--per-label-counts",
        default="",
        help="Per-label validation counts, e.g. 多车事故=70,拥堵=80,异常停车=80.",
    )
    ap.add_argument("--min-per-label", type=int, default=20)
    ap.add_argument("--max-per-label", type=int, default=120)
    args = ap.parse_args()

    labels = parse_csv(args.labels)
    rows = [normalize_row(row, i) for i, row in enumerate(read_jsonl(args.input_jsonl))]

    buckets = defaultdict(list)
    skipped = []
    for row in rows:
        label = detect_label(row, labels)
        if not label:
            skipped.append(row)
            continue
        row["_split_label"] = label
        buckets[label].append(row)

    rng = random.Random(args.seed)
    val_ids = set()
    summary = {}
    per_label_counts = parse_label_counts(args.per_label_counts)
    for label in labels:
        items = buckets[label]
        rng.shuffle(items)
        if label in per_label_counts:
            n_val = min(per_label_counts[label], len(items))
        elif args.per_label > 0:
            n_val = min(args.per_label, len(items))
        else:
            n_val = int(round(len(items) * args.val_ratio))
            n_val = max(args.min_per_label, n_val) if items else 0
            n_val = min(args.max_per_label, n_val, len(items))
        for row in items[:n_val]:
            val_ids.add(id(row))
        summary[label] = {"total": len(items), "val": n_val, "train": len(items) - n_val}

    val_rows = []
    train_rows = []
    for label in labels:
        for row in buckets[label]:
            clean = dict(row)
            clean.pop("_split_label", None)
            if id(row) in val_ids:
                val_rows.append(clean)
            else:
                train_rows.append(clean)

    write_jsonl(args.val_out, val_rows)
    if args.train_out:
        write_jsonl(args.train_out, train_rows)

    print("input:", args.input_jsonl)
    print("val_out:", args.val_out, "n=", len(val_rows))
    if args.train_out:
        print("train_out:", args.train_out, "n=", len(train_rows))
    print("label_summary:", json.dumps(summary, ensure_ascii=False, indent=2))
    print("skipped_unlabeled:", len(skipped))
    print("val_dist:", dict(Counter(detect_label(row, labels) for row in val_rows)))


if __name__ == "__main__":
    main()
