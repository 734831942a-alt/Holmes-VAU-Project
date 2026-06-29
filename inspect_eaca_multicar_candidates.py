import argparse
import json
from pathlib import Path


TARGET = "多车事故"


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def by_video(rows):
    out = {}
    for row in rows:
        video = row.get("video")
        if video:
            out[video] = row
    return out


def label_from_video(video):
    return str(video or "").split("_")[0]


def compact_text(text, max_len):
    text = " ".join(str(text or "").split())
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def main():
    ap = argparse.ArgumentParser(description="Inspect EACA multicar candidates and clip-level raw outputs.")
    ap.add_argument("--eaca", required=True)
    ap.add_argument("--base-pred", default="")
    ap.add_argument("--base-pred-field", default="pred")
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--max-raw-chars", type=int, default=500)
    args = ap.parse_args()

    eaca_rows = read_jsonl(args.eaca)
    base_map = by_video(read_jsonl(args.base_pred)) if args.base_pred else {}

    n = 0
    for row in eaca_rows:
        if row.get("eaca_label") != args.target:
            continue
        n += 1
        video = row.get("video", "")
        base = base_map.get(video, {})
        print("=" * 100)
        print("video:", video)
        print("gt_label:", row.get("gt_label") or label_from_video(video))
        print("base_pred:", compact_text(base.get(args.base_pred_field, ""), 160))
        print("eaca_reason:", row.get("eaca_reason"))
        print("signal_counts:", row.get("signal_counts"))
        print("clip_label_counts:", row.get("clip_label_counts"))
        for clip in row.get("clips", []):
            print("-" * 80)
            print("clip_id:", clip.get("clip_id"), "frames:", clip.get("frame_indices"))
            print("parsed:", clip.get("parsed"))
            print("raw:", compact_text(clip.get("raw", ""), args.max_raw_chars))

    print("=" * 100)
    print("total_candidates:", n)


if __name__ == "__main__":
    main()
