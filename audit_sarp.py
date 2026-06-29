import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_TARGETS = "多车事故,拥堵,异常停车,占道施工,二轮车辆闯入,抛洒物"


def parse_csv(text):
    return [x.strip() for x in (text or "").split(",") if x.strip()]


def load_labels(path):
    labels = {}
    totals = Counter()
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            video = obj.get("video", "")
            label = video.split("_")[0] if "_" in video else "UNKNOWN"
            labels[video] = label
            totals[label] += 1
    return labels, totals


def main():
    ap = argparse.ArgumentParser(description="Audit SARP road-surface-object signals.")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--ovd", required=True)
    ap.add_argument("--targets", default=DEFAULT_TARGETS)
    ap.add_argument("--examples", type=int, default=8)
    args = ap.parse_args()

    targets = parse_csv(args.targets)
    labels, totals = load_labels(args.gt)
    stats = defaultdict(Counter)
    suppression_counts = defaultdict(Counter)
    label_hits = defaultdict(Counter)
    examples = defaultdict(list)

    with Path(args.ovd).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            video = obj.get("video", "")
            label = labels.get(video, video.split("_")[0] if "_" in video else "UNKNOWN")
            if targets and label not in targets:
                continue

            sarp = ((obj.get("ovd") or {}).get("sarp") or {})
            if not sarp.get("ok"):
                stats[label]["not_ok"] += 1
                continue

            stats[label]["ok"] += 1
            if sarp.get("raw_triggered"):
                stats[label]["raw_triggered"] += 1
            if sarp.get("triggered"):
                stats[label]["triggered"] += 1
            else:
                stats[label]["none"] += 1

            if sarp.get("suppression_reason"):
                suppression_counts[label][sarp.get("suppression_reason")] += 1

            for k, v in (sarp.get("label_hits") or {}).items():
                label_hits[label][k] += v

            if sarp.get("raw_triggered") and len(examples[label]) < args.examples:
                examples[label].append({
                    "video": video,
                    "triggered": sarp.get("triggered"),
                    "raw_triggered": sarp.get("raw_triggered"),
                    "suppression_reason": sarp.get("suppression_reason"),
                    "candidate_count": sarp.get("candidate_count"),
                    "evidence_frames": sarp.get("evidence_frames"),
                    "label_hits": sarp.get("label_hits"),
                    "semantic_peak": sarp.get("semantic_peak"),
                    "rejected": sarp.get("rejected"),
                    "examples": sarp.get("examples"),
                })

    print("类别               总数   SARP_OK  raw_trig  triggered  none")
    print("-" * 68)
    for label in targets:
        c = stats[label]
        total = totals.get(label, 0)
        ok = c.get("ok", 0)
        print(
            f"{label:<12} {total:6d} {ok:8d} "
            f"{c.get('raw_triggered', 0):9d} "
            f"{c.get('triggered', 0):10d} "
            f"{c.get('none', 0):5d}"
        )

    print("\n=== rates within SARP_OK ===")
    for label in targets:
        ok = max(stats[label].get("ok", 0), 1)
        print(
            label,
            {
                "raw": round(stats[label].get("raw_triggered", 0) / ok, 3),
                "triggered": round(stats[label].get("triggered", 0) / ok, 3),
            },
        )

    print("\n=== suppression counts ===")
    for label in targets:
        if suppression_counts[label]:
            print(label, dict(suppression_counts[label]))

    print("\n=== label hits ===")
    for label in targets:
        if label_hits[label]:
            print(label, dict(label_hits[label]))

    print("\n=== examples ===")
    for label in targets:
        if examples[label]:
            print("\n", label)
            for item in examples[label]:
                print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
