import argparse
import csv
import json
import re
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
证据不足时不要猜测为“多车事故”，选择更具体、更保守的类别。"""

EVIDENCE_BY_REASON = {
    "collision_visible": ("可见碰撞过程", "画面中可见两辆及以上机动车发生碰撞、追尾或剐蹭过程。"),
    "collision_weak": ("疑似碰撞过程", "画面中存在疑似轻微碰撞、追尾或剐蹭迹象，但过程不够清晰。"),
    "damage_trace": ("事故痕迹证据", "画面中可见车辆损伤、碰撞痕迹或事故遗留状态。"),
    "post_incident_handling": ("事故后处置证据", "画面中可见事故后车辆滞留，人员在车辆附近查看、交涉或处置。"),
    "post_incident_handling_with_evidence": ("事故后处置证据", "画面中可见事故后车辆滞留、人员查看或交涉，并伴随明确车辆关系或现场异常证据。"),
    "post_incident_handling_visual_weak": ("弱事故后处置证据", "画面中可见人员在车辆附近查看或处置，但事故关系较弱，需要结合车辆位置和交通影响判断。"),
    "far_but_handling": ("远距离事故后处置证据", "事故车辆距离较远或画面不够清晰，但可见人员下车查看、交涉等事故后处置行为。"),
    "weak_evidence": ("弱事故证据", "画面存在事故后异常线索，但证据强度较弱，需要保守判断。"),
    "single_vehicle_but_evidence": ("单车事故相关证据", "画面主要为单车状态，但存在事故痕迹或处置证据。"),
}


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


def label_from_video(video: str) -> str:
    name = normalize_video_name(video)
    prefix = name.split("_", 1)[0] if "_" in name else Path(name).stem
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


def should_inject_boundary(label: str, scope: str) -> bool:
    if scope == "all":
        return True
    if scope == "target-only":
        return label == TARGET_LABEL
    if scope == "target-hard-negatives":
        return label == TARGET_LABEL or label in HARD_NEGATIVE_LABELS
    raise ValueError(f"unknown constraint scope: {scope}")


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


def inject_boundary(row: dict[str, Any], scope: str, constraint: str) -> tuple[dict[str, Any], bool]:
    out = json.loads(json.dumps(row, ensure_ascii=False))
    label = extract_label(out)
    if not should_inject_boundary(label, scope):
        out["multicar_boundary_injected"] = False
        out["multicar_boundary_scope"] = scope
        return out, False

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

    out["multicar_boundary_injected"] = True
    out["multicar_boundary_scope"] = scope
    return out, changed


def load_review_map(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            event_id = str(row.get("event_id") or "").strip()
            if event_id:
                out[event_id] = {k: str(v or "").strip() for k, v in row.items()}
    return out


def find_review(row: dict[str, Any], review_map: dict[str, dict[str, str]]) -> dict[str, str] | None:
    for event_id in event_ids_from_video(str(row.get("video") or row.get("image") or "")):
        if event_id in review_map:
            return review_map[event_id]
    return None


def evidence_text(review: dict[str, str]) -> tuple[str, str]:
    reason = review.get("human_reason", "")
    evidence_type, desc = EVIDENCE_BY_REASON.get(
        reason,
        ("事故后可见证据", "人工复核认为该样本存在可支持多车事故判断的可见证据。"),
    )
    note = review.get("human_note") or review.get("manual_note") or ""
    note = re.sub(r"\s+", " ", note).strip()
    if note:
        desc = f"{desc} 人工标注备注：{note}"
    return evidence_type, desc


def inject_evidence_to_answer(row: dict[str, Any], review: dict[str, str]) -> tuple[dict[str, Any], bool]:
    out = json.loads(json.dumps(row, ensure_ascii=False))
    evidence_type, desc = evidence_text(review)
    evidence_line = f"事故证据类型：{evidence_type}。证据说明：{desc}"

    conversations = out.get("conversations")
    if isinstance(conversations, list):
        for turn in conversations:
            if isinstance(turn, dict) and turn.get("from") in {"gpt", "assistant"}:
                old = str(turn.get("value", "") or "")
                if "事故证据类型：" not in old:
                    turn["value"] = evidence_line + "\n" + old
                    out["manual_evidence_injected"] = True
                    out["manual_evidence_reason"] = review.get("human_reason", "")
                    return out, True
                return out, False

    if out.get("answer") or out.get("anwser"):
        key = "answer" if out.get("answer") else "anwser"
        old = str(out.get(key) or "")
        if "事故证据类型：" not in old:
            out[key] = evidence_line + "\n" + old
            out["manual_evidence_injected"] = True
            out["manual_evidence_reason"] = review.get("human_reason", "")
            return out, True

    out["manual_evidence_injected"] = False
    return out, False


def transform_file(
    input_path: Path,
    output_path: Path,
    review_map: dict[str, dict[str, str]],
    boundary_scope: str,
    inject_evidence: bool,
    constraint: str,
) -> dict[str, Any]:
    rows = read_jsonl(input_path)
    out_rows = []
    labels = Counter()
    boundary_count = 0
    evidence_count = 0
    evidence_reason = Counter()
    review_found = 0
    review_yes = 0
    unknown_label = 0

    for row in rows:
        label = extract_label(row)
        if label:
            labels[label] += 1
        else:
            unknown_label += 1

        out, did_boundary = inject_boundary(row, boundary_scope, constraint)
        if did_boundary:
            boundary_count += 1

        review = find_review(row, review_map)
        if review:
            review_found += 1
        if (
            inject_evidence
            and label == TARGET_LABEL
            and review
            and review.get("human_keep") == "yes"
        ):
            review_yes += 1
            out, did_evidence = inject_evidence_to_answer(out, review)
            if did_evidence:
                evidence_count += 1
                evidence_reason[review.get("human_reason", "")] += 1
        else:
            out.setdefault("manual_evidence_injected", False)

        out_rows.append(out)

    write_jsonl(output_path, out_rows)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "total": len(out_rows),
        "labels": dict(labels),
        "unknown_label": unknown_label,
        "boundary_injected": boundary_count,
        "review_found": review_found,
        "review_yes": review_yes,
        "manual_evidence_injected": evidence_count,
        "manual_evidence_reason": dict(evidence_reason),
    }


def split_train_by_class(train_jsonl: Path, by_class_dir: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(train_jsonl)
    by_class: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = extract_label(row) or "unknown"
        by_class.setdefault(label, []).append(row)

    by_class_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for label, label_rows in sorted(by_class.items()):
        out_path = by_class_dir / f"{label}.jsonl"
        write_jsonl(out_path, label_rows)
        written[label] = {"annotation": str(out_path.resolve()).replace("\\", "/"), "length": len(label_rows)}
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
    root = str(video_root.resolve()).replace("\\", "/")
    meta = {}
    for label, info in class_files.items():
        repeat = repeat_default
        if label == TARGET_LABEL:
            repeat = repeat_multicar
        elif label == "抛洒物":
            repeat = repeat_debris
        meta[label] = {
            "root": root,
            "annotation": info["annotation"],
            "data_augment": False,
            "repeat_time": repeat,
            "length": info["length"],
        }
    meta_output.parent.mkdir(parents=True, exist_ok=True)
    meta_output.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build E9 evidence-supervised JSONL files from manual multi-car review labels."
    )
    p.add_argument("--review-csv", required=True)
    p.add_argument("--train-input", required=True)
    p.add_argument("--train-output", required=True)
    p.add_argument("--test-input")
    p.add_argument("--test-output")
    p.add_argument(
        "--boundary-scope",
        choices=["all", "target-only", "target-hard-negatives"],
        default="target-hard-negatives",
    )
    p.add_argument("--constraint-file")
    p.add_argument("--no-train-evidence", action="store_true")
    p.add_argument("--meta-output")
    p.add_argument("--video-root")
    p.add_argument("--by-class-dir")
    p.add_argument("--repeat-multicar", type=int, default=8)
    p.add_argument("--repeat-debris", type=int, default=1)
    p.add_argument("--repeat-default", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    review_map = load_review_map(Path(args.review_csv))
    constraint = (
        Path(args.constraint_file).read_text(encoding="utf-8")
        if args.constraint_file
        else MULTICAR_BOUNDARY_CONSTRAINT
    )
    if BOUNDARY_MARKER not in constraint:
        raise SystemExit(f"constraint must contain marker: {BOUNDARY_MARKER}")

    summaries = []
    summaries.append(
        transform_file(
            Path(args.train_input),
            Path(args.train_output),
            review_map,
            args.boundary_scope,
            not args.no_train_evidence,
            constraint,
        )
    )
    if args.test_input:
        if not args.test_output:
            raise SystemExit("--test-output is required with --test-input.")
        summaries.append(
            transform_file(
                Path(args.test_input),
                Path(args.test_output),
                review_map,
                args.boundary_scope,
                False,
                constraint,
            )
        )

    meta = None
    if args.meta_output:
        if not args.video_root:
            raise SystemExit("--meta-output requires --video-root.")
        by_class_dir = Path(args.by_class_dir) if args.by_class_dir else Path(args.train_output).parent / "by_class_train_e9_evidence"
        meta = build_meta(
            Path(args.train_output),
            Path(args.meta_output),
            Path(args.video_root),
            by_class_dir,
            args.repeat_multicar,
            args.repeat_debris,
            args.repeat_default,
        )

    print(json.dumps({"review_csv": args.review_csv, "summaries": summaries, "meta": meta}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
