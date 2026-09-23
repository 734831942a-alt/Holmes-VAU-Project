"""Deterministic stratified sampler for SPEC-02 batch 1."""

import argparse
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List

import cv2
import yaml


def media_info(path: Path) -> Dict:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    if fps <= 0 or frames <= 0 or width <= 0 or height <= 0:
        raise ValueError("invalid media metadata")
    return {"duration": frames / fps, "fps": fps, "width": width, "height": height}


def duration_summary(values: List[float]) -> Dict:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def scan(config: Dict):
    root = Path(config["data_root"])
    pattern = re.compile(config["filename_pattern"])
    categories = list(config["categories"])
    allowed = set(categories)
    night_hours = {int(value) for value in config["night_hours"]}
    minimum = float(config["min_duration_seconds"])
    maximum = float(config["max_duration_seconds"])
    eligible = {category: [] for category in categories}
    exclusions = Counter()

    for path in sorted(root.glob("*.mp4")):
        match = pattern.match(path.name)
        if not match:
            exclusions["filename_pattern_mismatch"] += 1
            continue
        category = match.group("category")
        if category not in allowed:
            exclusions["unknown_category"] += 1
            continue
        try:
            info = media_info(path)
        except ValueError:
            exclusions["invalid_fixed_media"] += 1
            continue
        if info["duration"] < minimum:
            exclusions["duration_below_min"] += 1
            continue
        if info["duration"] > maximum:
            exclusions["duration_above_max"] += 1
            continue
        timestamp = match.group("timestamp")
        eligible[category].append({
            "video_id": path.stem,
            "path": str(path),
            "category": category,
            "duration": round(info["duration"], 3),
            "fps": round(info["fps"], 3),
            "width": info["width"],
            "height": info["height"],
            "timestamp": timestamp,
            "is_night": int(timestamp[8:10]) in night_hours,
            "batch": config["batch"],
        })

    source_names = {path.name for path in Path(config["source_root"]).glob("*.mp4")}
    fixed_names = {path.name for path in root.glob("*.mp4")}
    exclusions["damaged_source_without_fixed_counterpart"] = len(source_names - fixed_names)
    exclusions["derived_copies_outside_sampling_root"] = sum(
        1 for derived in config["excluded_derived_roots"] for _ in Path(derived).rglob("*.mp4")
    )
    return eligible, exclusions


def choose(eligible: Dict[str, List[Dict]], config: Dict) -> List[Dict]:
    rng = random.Random(int(config["seed"]))
    target = int(config["per_category"])
    required_night = int(config["min_night_per_category"])
    selected = []
    for category in config["categories"]:
        pool = list(eligible[category])
        nights = [row for row in pool if row["is_night"]]
        actual_target = min(target, len(pool))
        night_count = min(required_night, actual_target, len(nights))
        chosen = rng.sample(nights, night_count)
        chosen_ids = {row["video_id"] for row in chosen}
        remaining = [row for row in pool if row["video_id"] not in chosen_ids]
        chosen.extend(rng.sample(remaining, actual_target - len(chosen)))
        selected.extend(chosen)
    return selected


def write_manifest(rows: List[Dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(selected, eligible, exclusions, config, output: Path) -> None:
    lines = [
        "# SPEC-02 Batch 1 分层抽样报告", "",
        f"- seed: {config['seed']}",
        f"- 数据源：{config['data_root']}",
        f"- 时长约束：{config['min_duration_seconds']}s <= duration <= {config['max_duration_seconds']}s",
        f"- 每类目标：{config['per_category']}，每类夜间至少 {config['min_night_per_category']}",
        f"- 夜间小时：{config['night_hours']}",
        f"- 实际总数：**{len(selected)}**", "",
        "## 各类结果", "",
        "| 类别 | 约束内候选 | 抽取数 | 夜间数 | 时长 min / median / max（秒） |",
        "|---|---:|---:|---:|---:|",
    ]
    shortages = []
    for category in config["categories"]:
        rows = [row for row in selected if row["category"] == category]
        nights = sum(row["is_night"] for row in rows)
        stats = duration_summary([row["duration"] for row in rows])
        lines.append(
            f"| {category} | {len(eligible[category])} | {len(rows)} | {nights} | "
            f"{stats['min']:.3f} / {stats['median']:.3f} / {stats['max']:.3f} |"
        )
        if len(rows) < int(config["per_category"]):
            shortages.append(f"- {category} 仅有 {len(eligible[category])} 个满足全部约束的候选，按实际数量抽取。")
        if nights < int(config["min_night_per_category"]):
            shortages.append(f"- {category} 仅抽到 {nights} 个夜间候选；未放宽约束补齐。")
    lines.extend(["", "## 不足说明", ""])
    lines.extend(shortages or ["六类均满足每类 10 段且夜间至少 3 段。"])
    lines.extend(["", "## 排除统计", ""])
    for reason in sorted(exclusions):
        lines.append(f"- {reason}: {exclusions[reason]}")
    lines.extend(["", "说明：抽样宇宙严格限定为 video_fixed/ 根目录；预览和案例目录仅计数、从未进入候选池。damaged_source_without_fixed_counterpart 是 video/ 中无同名 fixed 文件的损坏源条目。", ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def run(config_path: Path, manifest_override: str = "", report_override: str = "") -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    eligible, exclusions = scan(config)
    selected = choose(eligible, config)
    write_manifest(selected, Path(manifest_override or config["manifest_path"]))
    write_report(selected, eligible, exclusions, config, Path(report_override or config["report_path"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--report-out", default="")
    args = parser.parse_args()
    run(Path(args.config), args.manifest_out, args.report_out)


if __name__ == "__main__":
    main()
