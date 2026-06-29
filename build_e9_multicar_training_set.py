import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


LABELS = [
    "多车事故",
    "拥堵",
    "异常停车",
    "占道施工",
    "二轮车辆闯入",
    "抛洒物",
]

TARGET_LABEL = "多车事故"
HARD_NEGATIVE_LABELS = {"拥堵", "异常停车", "二轮车辆闯入"}

BOUNDARY_MARKER = "[多车事故判别边界]"
MULTICAR_BOUNDARY_CONSTRAINT = """[多车事故判别边界]
请把“多车事故”当作高证据门槛类别处理。
只有明确看到两辆及以上机动车发生碰撞、追尾、事故后共同滞留，或看到事故现场处置证据时，才判断为“多车事故”。
如果只是单车静止、单车占道、单车停在应急车道或行车道，优先判断为“异常停车”。
如果是大范围车辆排队、整体低速、密集缓行或多车同时走走停停，优先判断为“拥堵”。
如果画面核心异常是摩托车、电动车或非机动车进入主路，优先判断为“二轮车辆闯入”。
证据不足时不要猜测为“多车事故”，选择更具体、更保守的类别。
"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def label_from_video(video: str) -> str:
    name = normalize_video_name(video)
    if "_" in name:
        prefix = name.split("_", 1)[0]
    else:
        prefix = Path(name).stem
    return prefix if prefix in LABELS else ""


def extract_label(row: dict[str, Any]) -> str:
    for key in ("gt", "label", "category", "class", "cls"):
        value = row.get(key)
        if isinstance(value, str) and value in LABELS:
            return value
    for key in ("video", "image"):
        value = row.get(key)
        if value:
            label = label_from_video(str(value))
            if label:
                return label
    return ""


def should_inject(label: str, scope: str) -> bool:
    if scope == "all":
        return True
    if scope == "target-only":
        return label == TARGET_LABEL
    if scope == "target-hard-negatives":
        return label == TARGET_LABEL or label in HARD_NEGATIVE_LABELS
    raise ValueError(f"unknown constraint scope: {scope}")


def inject_constraint(text: str, constraint: str) -> tuple[str, bool]:
    text = "" if text is None else str(text)
    if BOUNDARY_MARKER in text:
        return text, False

    # Keep the special video token first; InternVL examples usually start with it.
    stripped = text.lstrip()
    leading = text[: len(text) - len(stripped)]
    if stripped.startswith("<video>"):
        after_token = stripped[len("<video>") :].lstrip("\r\n")
        return leading + "<video>\n" + constraint.rstrip() + "\n" + after_token, True
    if stripped.startswith("<image>"):
        after_token = stripped[len("<image>") :].lstrip("\r\n")
        return leading + "<image>\n" + constraint.rstrip() + "\n" + after_token, True
    return constraint.rstrip() + "\n" + text.lstrip(), True


def update_row(row: dict[str, Any], constraint: str, scope: str) -> tuple[dict[str, Any], bool, str]:
    out = json.loads(json.dumps(row, ensure_ascii=False))
    label = extract_label(out)
    if not should_inject(label, scope):
        out["multicar_boundary_injected"] = False
        out["multicar_boundary_scope"] = scope
        return out, False, label

    changed = False
    if out.get("prompt"):
        out["prompt"], c = inject_constraint(str(out.get("prompt", "")), constraint)
        changed = changed or c

    conversations = out.get("conversations")
    if isinstance(conversations, list):
        for turn in conversations:
            if isinstance(turn, dict) and turn.get("from") == "human":
                turn["value"], c = inject_constraint(str(turn.get("value", "")), constraint)
                changed = changed or c
                break

    out["multicar_boundary_injected"] = True
    out["multicar_boundary_scope"] = scope
    return out, changed, label


def transform_file(input_path: Path, output_path: Path, constraint: str, scope: str) -> dict[str, Any]:
    rows = read_jsonl(input_path)
    out_rows: list[dict[str, Any]] = []
    labels = Counter()
    injected = 0
    changed = 0
    unknown_label = 0
    for row in rows:
        out, did_change, label = update_row(row, constraint, scope)
        out_rows.append(out)
        if label:
            labels[label] += 1
        else:
            unknown_label += 1
        if out.get("multicar_boundary_injected"):
            injected += 1
        if did_change:
            changed += 1

    write_jsonl(output_path, out_rows)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "total": len(out_rows),
        "injected": injected,
        "changed": changed,
        "unknown_label": unknown_label,
        "labels": dict(labels),
    }


def split_train_by_class(train_jsonl: Path, by_class_dir: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(train_jsonl)
    by_class: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = extract_label(row) or "unknown"
        by_class.setdefault(label, []).append(row)

    by_class_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, dict[str, Any]] = {}
    for label, label_rows in sorted(by_class.items()):
        out_path = by_class_dir / f"{label}.jsonl"
        write_jsonl(out_path, label_rows)
        written[label] = {
            "annotation": str(out_path.resolve()).replace("\\", "/"),
            "length": len(label_rows),
        }
    return written


def build_meta(
    train_jsonl: Path,
    meta_output: Path,
    video_root: Path,
    by_class_dir: Path,
    repeat_multicar: int,
    repeat_debris: int,
    repeat_default: int,
) -> dict[str, Any]:
    class_files = split_train_by_class(train_jsonl, by_class_dir)
    video_root_str = str(video_root.resolve()).replace("\\", "/")
    meta: dict[str, Any] = {}
    for label, info in class_files.items():
        repeat = repeat_default
        if label == "多车事故":
            repeat = repeat_multicar
        elif label == "抛洒物":
            repeat = repeat_debris
        meta[label] = {
            "root": video_root_str,
            "annotation": info["annotation"],
            "data_augment": False,
            "repeat_time": repeat,
            "length": info["length"],
        }
    meta_output.parent.mkdir(parents=True, exist_ok=True)
    meta_output.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build E9 train/test JSONL files with a consistent multi-car accident "
            "class-boundary constraint."
        )
    )
    parser.add_argument("--train-input", help="Original train JSONL.")
    parser.add_argument("--train-output", help="Output train JSONL with boundary constraint.")
    parser.add_argument("--test-input", help="Original test JSONL.")
    parser.add_argument("--test-output", help="Output test JSONL with boundary constraint.")
    parser.add_argument(
        "--constraint-scope",
        choices=["all", "target-only", "target-hard-negatives"],
        default="all",
        help="Which samples receive the boundary prompt. Use all for train/test distribution consistency.",
    )
    parser.add_argument(
        "--constraint-file",
        help="Optional UTF-8 text file. If omitted, use the built-in multi-car boundary constraint.",
    )
    parser.add_argument("--meta-output", help="Optional balanced meta JSON for InternVL training.")
    parser.add_argument("--video-root", help="Video root used in the meta JSON.")
    parser.add_argument("--by-class-dir", help="Optional output dir for class-split train JSONLs.")
    parser.add_argument("--repeat-multicar", type=int, default=8)
    parser.add_argument("--repeat-debris", type=int, default=9)
    parser.add_argument("--repeat-default", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.train_input and not args.test_input:
        raise SystemExit("Provide at least --train-input or --test-input.")

    constraint = MULTICAR_BOUNDARY_CONSTRAINT
    if args.constraint_file:
        constraint = Path(args.constraint_file).read_text(encoding="utf-8")
    if BOUNDARY_MARKER not in constraint:
        raise SystemExit(f"constraint must contain marker: {BOUNDARY_MARKER}")

    summaries = []
    if args.train_input:
        if not args.train_output:
            raise SystemExit("--train-output is required when --train-input is set.")
        summaries.append(
            transform_file(
                Path(args.train_input),
                Path(args.train_output),
                constraint,
                args.constraint_scope,
            )
        )
    if args.test_input:
        if not args.test_output:
            raise SystemExit("--test-output is required when --test-input is set.")
        summaries.append(
            transform_file(
                Path(args.test_input),
                Path(args.test_output),
                constraint,
                args.constraint_scope,
            )
        )

    meta = None
    if args.meta_output:
        if not args.train_output:
            raise SystemExit("--meta-output requires --train-output.")
        if not args.video_root:
            raise SystemExit("--meta-output requires --video-root.")
        by_class_dir = (
            Path(args.by_class_dir)
            if args.by_class_dir
            else Path(args.train_output).parent / "by_class_train_e9_multicar_boundary"
        )
        meta = build_meta(
            Path(args.train_output),
            Path(args.meta_output),
            Path(args.video_root),
            by_class_dir,
            args.repeat_multicar,
            args.repeat_debris,
            args.repeat_default,
        )

    print(json.dumps({"summaries": summaries, "meta": meta}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
