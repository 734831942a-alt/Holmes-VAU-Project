import argparse
import json
from pathlib import Path


def load_jsonl(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def save_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build a small debug subset for InternVL video fine-tuning."
    )
    parser.add_argument(
        "--source-jsonl",
        default=r"E:\高架桥数据\traffic_train.jsonl",
        help="Path to the full training jsonl.",
    )
    parser.add_argument(
        "--video-root",
        default=r"E:\高架桥数据\video",
        help="Root folder containing video files.",
    )
    parser.add_argument(
        "--out-dir",
        default="internvl_chat/shell",
        help="Where to write debug jsonl/meta.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of valid samples to keep.",
    )
    args = parser.parse_args()

    source_jsonl = Path(args.source_jsonl)
    video_root = Path(args.video_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(source_jsonl)
    picked = []
    for row in rows:
        rel = row.get("video", "")
        video_path = video_root / rel
        if video_path.exists():
            row["id"] = len(picked)
            picked.append(row)
        if len(picked) >= args.num_samples:
            break

    if not picked:
        raise RuntimeError("No valid samples found. Check source jsonl and video-root.")

    debug_jsonl = out_dir / "traffic_train_debug.jsonl"
    save_jsonl(debug_jsonl, picked)

    meta = {
        "HIVAU_debug": {
            "root": str(video_root).replace("\\", "/"),
            "annotation": str(debug_jsonl.resolve()).replace("\\", "/"),
            "data_augment": False,
            "repeat_time": 1,
            "length": len(picked),
        }
    }
    debug_meta = out_dir / "holmesvau_data_debug.json"
    debug_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"source: {source_jsonl}")
    print(f"video root: {video_root}")
    print(f"picked samples: {len(picked)}")
    print(f"debug jsonl: {debug_jsonl}")
    print(f"debug meta: {debug_meta}")


if __name__ == "__main__":
    main()
