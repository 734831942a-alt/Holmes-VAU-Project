"""Camera/source-event connected components are indivisible; test membership is locked."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from fractions import Fraction
import random
import re

from data.common import config, read_jsonl, write_json, write_jsonl, write_text, issue, fingerprint
from data.cameras import Components


def event_key(row, cfg):
    match = re.fullmatch(cfg["source_event"]["regex"], row["video_id"])
    if match is None:
        return None
    return cfg["source_event"]["separator"].join(match.group(field) for field in cfg["source_event"]["fields"])


def leakage(rows, splits, cfg):
    membership = {}
    for name, ids in splits.items():
        for video_id in ids:
            if video_id in membership:
                raise ValueError("Duplicate split membership")
            membership[video_id] = name
    if set(membership) != {r["video_id"] for r in rows}:
        raise ValueError("Split coverage mismatch")
    cameras, events = defaultdict(set), defaultdict(set)
    for row in rows:
        cameras[row["camera_group"]].add(membership[row["video_id"]])
        key = event_key(row, cfg)
        if key is not None:
            events[key].add(membership[row["video_id"]])
    return {"cross_split_leak": sum(len(v) > 1 for v in cameras.values()),
            "source_event_cross_split_leak": sum(len(v) > 1 for v in events.values())}


def assign(rows, cfg, locked_test=None):
    ids = [r["video_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate video IDs")
    graph = Components(ids)
    cameras, events = {}, {}
    for row in rows:
        vid = row["video_id"]
        group = row["camera_group"]
        if group in cameras:
            graph.join(vid, cameras[group])
        cameras[group] = vid
        key = event_key(row, cfg)
        if key is not None:
            if key in events:
                graph.join(vid, events[key])
            events[key] = vid
    components = defaultdict(list)
    for vid in ids:
        components[graph.root(vid)].append(vid)
    blocks = list(components.values())
    rng = random.Random(cfg["seed"])
    rng.shuffle(blocks)
    blocks.sort(key=len, reverse=True)
    ratios = cfg["split_ratios"]
    if sum(Fraction(str(v)) for v in ratios.values()) != 1 or set(ratios) != {"train", "val", "test"}:
        raise ValueError("Invalid split ratios")
    splits = {name: [] for name in ratios}
    if locked_test is not None:
        locked = set(locked_test)
        if not locked <= set(ids):
            raise ValueError("Locked test members missing from manifest; explicit --reseed required")
        for block in blocks:
            intersect = locked.intersection(block)
            if intersect and not set(block) <= locked:
                raise ValueError("Locked test conflicts with camera/source-event component; explicit --reseed required")
        splits["test"] = sorted(locked)
        blocks = [b for b in blocks if not locked.intersection(b)]
    choices = [name for name in ratios if locked_test is None or name != "test"]
    for block in blocks:
        # Greedy squared target-count error on indivisible components, with deterministic tie handling.
        name = min(choices, key=lambda k: (len(splits[k]) + len(block) - len(rows) * ratios[k]) ** 2 -
                                         (len(splits[k]) - len(rows) * ratios[k]) ** 2)
        splits[name].extend(block)
    splits = {name: sorted(values) for name, values in splits.items()}
    checks = leakage(rows, splits, cfg)
    if any(checks.values()):
        raise AssertionError(checks)
    return splits, checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reseed", action="store_true")
    args = parser.parse_args()
    cfg = config(args.config)
    rows = read_jsonl(args.manifest)
    lock_path = Path(cfg["split_lock"])
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else None
    previous = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else None
    if not args.reseed and lock is not None and previous is not None and previous["test"] != lock["test"]:
        raise ValueError("Existing test split differs from persisted lock")
    locked_test = None if args.reseed else (lock["test"] if lock else previous["test"] if previous else None)
    try:
        splits, checks = assign(rows, cfg, locked_test)
    except ValueError as exc:
        issue(cfg, "split-lock-conflict", "防泄漏与 test 锁冲突", str(exc))
        raise
    membership = {vid: name for name, vids in splits.items() for vid in vids}
    for row in rows:
        row["split"] = membership[row["video_id"]]
    counts = {category: {name: sum(r["category"] == category and r["split"] == name for r in rows)
                         for name in splits} for category in cfg["baseline_categories"]}
    table = "# SPEC-04 六类 × 三 split\n\n| 类别 | train | val | test |\n|---|---:|---:|---:|\n"
    for category, values in counts.items():
        table += "| " + category + " | " + " | ".join(str(values[n]) for n in splits) + " |\n"
    table += "\n```json\n" + json.dumps(checks, indent=2) + "\n```\n"
    table += "划分单位：camera_group 与可识别源事件的连通分量。无法解析来源的文件仅按 camera_group。\n"
    table += f"无法解析源事件的文件数：{sum(event_key(r, cfg) is None for r in rows)}。\n"
    empty = [(c, s) for c in cfg["rare_categories"] for s in splits if counts[c][s] == 0]
    if empty:
        issue(cfg, "rare-class-empty", "稀有类别在部分 split 为零", str(empty) + "\n需作者决定是否分层抽样；不自动改变划分。")
    write_text(cfg["split_table"], table)
    write_json(args.out, splits)
    if lock is None or args.reseed:
        write_json(lock_path, {"test": splits["test"], "seed": cfg["seed"], "config_sha256": fingerprint(cfg)})
    write_jsonl(args.manifest, rows)
    print(json.dumps(dict(checks, split_sizes={k: len(v) for k, v in splits.items()})), flush=True)


if __name__ == "__main__":
    main()
