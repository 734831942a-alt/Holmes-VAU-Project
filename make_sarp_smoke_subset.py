import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_PLAN = "抛洒物:60,占道施工:20,二轮车辆闯入:20,异常停车:15,拥堵:10,多车事故:10"


def extract_label(video):
    return video.split("_")[0] if video else ""


def parse_plan(text):
    plan = {}
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Bad plan item: {part!r}, expected label:number")
        label, value = part.split(":", 1)
        label = label.strip()
        value = int(value.strip())
        if value < 0:
            raise ValueError(f"Bad sample count for {label}: {value}")
        plan[label] = value
    return plan


def main():
    ap = argparse.ArgumentParser(description="Build a SARP-focused smoke subset.")
    ap.add_argument("--gt", required=True, help="Full train/test jsonl.")
    ap.add_argument("--out", required=True, help="Output smoke jsonl.")
    ap.add_argument("--plan", default=DEFAULT_PLAN, help=f"CSV plan, default: {DEFAULT_PLAN}")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    plan = parse_plan(args.plan)
    buckets = defaultdict(list)
    with Path(args.gt).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            label = extract_label(obj.get("video", ""))
            if label in plan:
                buckets[label].append(obj)

    rows = []
    print("=== SARP smoke sample plan ===")
    for label, wanted in plan.items():
        pool = buckets[label]
        picked = random.sample(pool, min(wanted, len(pool)))
        rows.extend(picked)
        print(f"{label}: available={len(pool)} sampled={len(picked)}")

    random.shuffle(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = Counter(extract_label(row.get("video", "")) for row in rows)
    print(f"\nOutput: {out}")
    print(f"Total: {len(rows)}")
    print("Counts:", dict(counts))


if __name__ == "__main__":
    main()
