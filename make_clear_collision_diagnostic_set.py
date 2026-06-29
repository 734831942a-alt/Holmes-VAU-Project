import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


BOUNDARY_MARKER = "[多车事故判别边界]"
MULTICAR_BOUNDARY_CONSTRAINT = """[多车事故判别边界]
请把“多车事故”当作高证据门槛类别处理。
只有明确看到两辆及以上机动车发生碰撞、追尾、事故后共同滞留，或看到事故现场处置证据时，才判断为“多车事故”。
如果只是单车静止、单车占道、单车停在应急车道或行车道，优先判断为“异常停车”。
如果是大范围车辆排队、整体低速、密集缓行或多车同时走走停停，优先判断为“拥堵”。
如果画面核心异常是摩托车、电动车或非机动车进入主路，优先判断为“二轮车辆闯入”。
证据不足时不要猜测为“多车事故”，选择更具体、更保守的类别。"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as w:
        for row in rows:
            w.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_video_name(video: str) -> str:
    return Path(str(video).replace("\\", "/")).name


def event_ids_from_video(video: str) -> list[str]:
    return re.findall(r"\d{6,}", Path(normalize_video_name(video)).stem)


def load_clear_collision_reviews(review_csv: Path) -> dict[str, dict[str, str]]:
    clear = {}
    with review_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            event_id = str(row.get("event_id") or "").strip()
            if not event_id:
                continue
            human_reason = str(row.get("human_reason") or "").strip()
            bucket = str(row.get("annotation_bucket") or "").strip()
            if human_reason == "collision_visible" or bucket == "collision_process_clear":
                clear[event_id] = {k: str(v or "").strip() for k, v in row.items()}
    return clear


def index_rows(paths: list[Path]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    by_event = {}
    source_by_event = {}
    for path in paths:
        for row in read_jsonl(path):
            for event_id in event_ids_from_video(str(row.get("video") or row.get("image") or "")):
                if event_id not in by_event:
                    by_event[event_id] = row
                    source_by_event[event_id] = str(path)
    return by_event, source_by_event


def inject_boundary_text(text: str, constraint: str) -> tuple[str, bool]:
    text = "" if text is None else str(text)
    if BOUNDARY_MARKER in text:
        return text, False
    stripped = text.lstrip()
    leading = text[: len(text) - len(stripped)]
    if stripped.startswith("<video>"):
        after_token = stripped[len("<video>") :].lstrip("\r\n")
        return leading + "<video>\n" + constraint.rstrip() + "\n" + after_token, True
    if stripped.startswith("<image>"):
        after_token = stripped[len("<image>") :].lstrip("\r\n")
        return leading + "<image>\n" + constraint.rstrip() + "\n" + after_token, True
    return constraint.rstrip() + "\n" + text.lstrip(), True


def inject_boundary(row: dict[str, Any], constraint: str) -> dict[str, Any]:
    out = json.loads(json.dumps(row, ensure_ascii=False))
    changed = False
    if out.get("prompt"):
        out["prompt"], c = inject_boundary_text(str(out.get("prompt", "")), constraint)
        changed = changed or c
    conversations = out.get("conversations")
    if isinstance(conversations, list):
        for turn in conversations:
            if isinstance(turn, dict) and turn.get("from") == "human":
                turn["value"], c = inject_boundary_text(str(turn.get("value", "")), constraint)
                changed = changed or c
                break
    out["multicar_boundary_injected"] = changed or (BOUNDARY_MARKER in json.dumps(out, ensure_ascii=False))
    out["diagnostic_boundary"] = True
    return out


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "event_id",
        "video",
        "source_jsonl",
        "idx",
        "human_reason",
        "annotation_bucket",
        "manual_note",
        "human_keep",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a clear-collision multi-car diagnostic JSONL set.")
    p.add_argument("--review-csv", required=True)
    p.add_argument("--input-jsonl", action="append", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--output-boundary-jsonl")
    p.add_argument("--manifest-csv")
    p.add_argument("--missing-txt")
    p.add_argument("--constraint-file")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    review_map = load_clear_collision_reviews(Path(args.review_csv))
    row_index, source_index = index_rows([Path(p) for p in args.input_jsonl])

    selected = []
    manifest = []
    missing = []
    for event_id, review in sorted(review_map.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
        row = row_index.get(event_id)
        if not row:
            missing.append(event_id)
            continue
        out = json.loads(json.dumps(row, ensure_ascii=False))
        out["clear_collision_diagnostic"] = True
        out["clear_collision_event_id"] = event_id
        out["clear_collision_human_reason"] = review.get("human_reason", "")
        out["clear_collision_manual_note"] = review.get("manual_note", "")
        selected.append(out)
        manifest.append(
            {
                "event_id": event_id,
                "video": normalize_video_name(str(row.get("video") or row.get("image") or "")),
                "source_jsonl": source_index.get(event_id, ""),
                "idx": review.get("idx", ""),
                "human_reason": review.get("human_reason", ""),
                "annotation_bucket": review.get("annotation_bucket", ""),
                "manual_note": review.get("manual_note", ""),
                "human_keep": review.get("human_keep", ""),
            }
        )

    write_jsonl(Path(args.output_jsonl), selected)
    if args.output_boundary_jsonl:
        constraint = (
            Path(args.constraint_file).read_text(encoding="utf-8")
            if args.constraint_file
            else MULTICAR_BOUNDARY_CONSTRAINT
        )
        if BOUNDARY_MARKER not in constraint:
            raise SystemExit(f"constraint must contain marker: {BOUNDARY_MARKER}")
        write_jsonl(Path(args.output_boundary_jsonl), [inject_boundary(row, constraint) for row in selected])
    if args.manifest_csv:
        write_manifest(Path(args.manifest_csv), manifest)
    if args.missing_txt:
        Path(args.missing_txt).parent.mkdir(parents=True, exist_ok=True)
        Path(args.missing_txt).write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")

    print(
        json.dumps(
            {
                "review_clear_collision_total": len(review_map),
                "found": len(selected),
                "missing": len(missing),
                "output_jsonl": args.output_jsonl,
                "output_boundary_jsonl": args.output_boundary_jsonl,
                "manifest_csv": args.manifest_csv,
                "missing_ids": missing,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
