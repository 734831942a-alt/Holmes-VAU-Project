"""Run the SPEC-01 B0 zero-shot temporal-localization baseline."""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

from src.model.holmesvau_infer import generate, load_model

STRICT_INTERVAL = re.compile(
    r"^\s*START\s*=\s*(\d+(?:\.\d+)?)\s*[,;]\s*END\s*=\s*(\d+(?:\.\d+)?)\s*$",
    re.IGNORECASE,
)


def parse_interval(text: str) -> Optional[Tuple[float, float]]:
    match = STRICT_INTERVAL.match(text)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def temporal_iou(pred: Tuple[float, float], gt: Tuple[float, float]) -> float:
    ps, pe = pred
    gs, ge = gt
    if pe <= ps or ge <= gs:
        return 0.0
    intersection = max(0.0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return intersection / union if union > 0 else 0.0


def load_segments(data_root: Path, annotation_sets: Iterable[Dict]) -> List[Dict]:
    segments = []
    for item in annotation_sets:
        path = data_root / item["path"]
        database = json.loads(path.read_text(encoding="utf-8"))
        for video_id, record in database.items():
            for event_index, boundary in enumerate(record.get("events") or []):
                segments.append(
                    {
                        "video_id": video_id,
                        "event_index": event_index,
                        "gt_start": float(boundary[0]),
                        "gt_end": float(boundary[1]),
                        "dataset": item["dataset"],
                        "split": item["split"],
                    }
                )
    return segments


def resolve_video(data_root: Path, segment: Dict, templates: Iterable[str]) -> Optional[Path]:
    values = dict(segment)
    for template in templates:
        candidate = data_root / template.format(**values)
        if candidate.is_file():
            return candidate
    return None


def summarize(predictions: List[Dict], frame_counts: List[int]) -> Dict:
    total = len(predictions)
    ious = [float(row["iou"]) for row in predictions]
    return {
        "miou": sum(ious) / total,
        "r1_iou_0.3": sum(value >= 0.3 for value in ious) / total,
        "r1_iou_0.5": sum(value >= 0.5 for value in ious) / total,
        "r1_iou_0.7": sum(value >= 0.7 for value in ious) / total,
        "parse_fail_rate": sum(not row["parse_ok"] for row in predictions) / total,
        "avg_frames": sum(frame_counts) / total,
    }


def run(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_root = Path(config["data_root"])
    output_dir = Path(config["output_dir"])
    segments = load_segments(data_root, config["annotation_sets"])
    count = int(config["num_samples"])
    if len(segments) < count:
        raise RuntimeError(f"only {len(segments)} event segments are available; need {count}")
    selected = random.Random(int(config["seed"])).sample(segments, count)

    missing = []
    for segment in selected:
        path = resolve_video(data_root, segment, config["video_templates"])
        segment["video_path"] = str(path) if path else None
        if path is None:
            missing.append(f'{segment["dataset"]}/{segment["split"]}/{segment["video_id"]}')
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"{len(missing)}/{count} sampled source videos are missing; first entries:\n{preview}"
        )

    model, tokenizer = load_model(config["model_path"], config["device"])
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    predictions: List[Dict] = []
    frame_counts: List[int] = []

    with predictions_path.open("w", encoding="utf-8") as handle:
        for segment in selected:
            raw, frame_indices = generate(
                video_path=segment["video_path"],
                prompt=config["prompt_template"],
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=int(config["max_new_tokens"]),
                temperature=float(config["temperature"]),
                num_frames=int(config["num_frames"]),
                input_size=int(config["input_size"]),
                max_tiles_per_frame=int(config["max_tiles_per_frame"]),
            )
            parsed = parse_interval(raw)
            parse_ok = parsed is not None
            if parsed is None:
                parsed_start = parsed_end = None
                iou = 0.0
            else:
                parsed_start, parsed_end = parsed
                iou = temporal_iou(
                    (parsed_start, parsed_end),
                    (segment["gt_start"], segment["gt_end"]),
                )
            row = {
                "video_id": f'{segment["video_id"]}#event-{segment["event_index"]}',
                "gt_start": segment["gt_start"],
                "gt_end": segment["gt_end"],
                "raw_output": raw,
                "parsed_start": parsed_start,
                "parsed_end": parsed_end,
                "iou": iou,
                "parse_ok": parse_ok,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            predictions.append(row)
            frame_counts.append(len(frame_indices))

    metrics = summarize(predictions, frame_counts)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(Path(args.config))


if __name__ == "__main__":
    main()
