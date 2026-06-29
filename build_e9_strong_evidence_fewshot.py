import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


LABELS = ["多车事故", "拥堵", "异常停车", "占道施工", "二轮车辆闯入", "抛洒物"]
TARGET_LABEL = "多车事故"
HARD_NEGATIVE_LABELS = {"拥堵", "异常停车", "二轮车辆闯入"}

BOUNDARY_MARKER = "[多车事故判别边界]"
BOUNDARY = """[多车事故判别边界]
请把“多车事故”当作高证据门槛类别处理。
只有出现以下证据之一时才判断为多车事故：
1. 明确可见两辆及以上机动车发生碰撞、追尾或剐蹭；
2. 未看到碰撞过程，但至少两辆机动车异常接近或共同滞留，且有人员在相关车辆之间查看、交涉或处置，并造成后方车辆绕行、减速或局部阻塞；
3. 可见车辆损伤、碰撞痕迹、事故处置或明显事故遗留状态。

以下情况不要判为多车事故：
1. 单车静止、单车占道、单车停靠且人员只在单车旁查看，优先判断为异常停车；
2. 大范围车辆排队、整体低速、密集缓行但无明确事故源，优先判断为拥堵；
3. 摩托车、电动车或非机动车进入主路，优先判断为二轮车辆闯入；
4. 仅有“可能碰撞风险”“后方车辆避让”等推测性描述，不足以判为多车事故。"""


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


def video_name(video: str) -> str:
    return Path(str(video).replace("\\", "/")).name


def event_ids(video: str) -> list[str]:
    return re.findall(r"\d{6,}", Path(video_name(video)).stem)


def gt_label(row: dict[str, Any]) -> str:
    video = str(row.get("video") or row.get("image") or "")
    name = video_name(video)
    prefix = name.split("_", 1)[0] if "_" in name else Path(name).stem
    return prefix if prefix in LABELS else ""


def row_key(row: dict[str, Any]) -> str:
    return video_name(str(row.get("video") or row.get("image") or ""))


def is_strong_review(row: dict[str, str]) -> bool:
    return row.get("human_reason") == "collision_visible" or row.get("annotation_bucket") == "collision_process_clear"


def load_review(path: Path) -> dict[str, dict[str, str]]:
    out = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row.get("event_id") or "").strip()
            if eid:
                out[eid] = {k: str(v or "").strip() for k, v in row.items()}
    return out


def index_rows(paths: list[Path]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    by_event = {}
    source = {}
    for path in paths:
        for row in read_jsonl(path):
            for eid in event_ids(str(row.get("video") or row.get("image") or "")):
                by_event.setdefault(eid, row)
                source.setdefault(eid, str(path))
    return by_event, source


def should_boundary(label: str) -> bool:
    return label == TARGET_LABEL or label in HARD_NEGATIVE_LABELS


def inject_text(text: str, block: str) -> tuple[str, bool]:
    text = "" if text is None else str(text)
    if BOUNDARY_MARKER in text:
        return text, False
    stripped = text.lstrip()
    leading = text[: len(text) - len(stripped)]
    if stripped.startswith("<video>"):
        after = stripped[len("<video>") :].lstrip("\r\n")
        return leading + "<video>\n" + block.rstrip() + "\n" + after, True
    return block.rstrip() + "\n" + text.lstrip(), True


def inject_boundary(row: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(row, ensure_ascii=False))
    if not should_boundary(gt_label(out)):
        out["multicar_boundary_injected"] = False
        return out
    changed = False
    if out.get("prompt"):
        out["prompt"], c = inject_text(str(out.get("prompt", "")), BOUNDARY)
        changed = changed or c
    if isinstance(out.get("conversations"), list):
        for turn in out["conversations"]:
            if isinstance(turn, dict) and turn.get("from") == "human":
                turn["value"], c = inject_text(str(turn.get("value", "")), BOUNDARY)
                changed = changed or c
                break
    out["multicar_boundary_injected"] = changed or (BOUNDARY_MARKER in json.dumps(out, ensure_ascii=False))
    return out


def inject_strong_evidence(row: dict[str, Any], review: dict[str, str]) -> dict[str, Any]:
    out = json.loads(json.dumps(row, ensure_ascii=False))
    note = re.sub(r"\s+", " ", review.get("manual_note", "")).strip()
    line = "事故证据类型：强证据多车事故。证据说明：人工复核标注为明确碰撞过程，可见两辆及以上机动车发生碰撞、追尾或剐蹭。"
    if note:
        line += f" 人工标注备注：{note}"
    if isinstance(out.get("conversations"), list):
        for turn in out["conversations"]:
            if isinstance(turn, dict) and turn.get("from") in {"gpt", "assistant"}:
                old = str(turn.get("value", "") or "")
                if "事故证据类型：" not in old:
                    turn["value"] = line + "\n" + old
                break
    elif out.get("answer") or out.get("anwser"):
        key = "answer" if out.get("answer") else "anwser"
        old = str(out.get(key) or "")
        if "事故证据类型：" not in old:
            out[key] = line + "\n" + old
    out["strong_evidence_train_injected"] = True
    return out


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["role", "event_id", "video", "source_jsonl", "human_reason", "annotation_bucket", "manual_note"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def split_strong_ids(
    strong_ids: list[str],
    source_by_event: dict[str, str],
    train_count: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    train_source = [eid for eid in strong_ids if "traffic_train" in source_by_event.get(eid, "")]
    other = [eid for eid in strong_ids if eid not in train_source]
    rng = random.Random(seed)
    rng.shuffle(train_source)
    rng.shuffle(other)
    train_ids = train_source[:train_count]
    if len(train_ids) < train_count:
        train_ids += other[: train_count - len(train_ids)]
    train_ids = sorted(train_ids)
    detect_ids = sorted([eid for eid in strong_ids if eid not in set(train_ids)])
    return train_ids, detect_ids


def build_meta(train_jsonl: Path, meta_output: Path, by_class_dir: Path, video_root: Path, repeat_multicar: int, repeat_debris: int) -> dict[str, Any]:
    rows = read_jsonl(train_jsonl)
    by_class: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_class.setdefault(gt_label(row) or "unknown", []).append(row)
    by_class_dir.mkdir(parents=True, exist_ok=True)
    meta = {}
    for label, label_rows in sorted(by_class.items()):
        out = by_class_dir / f"{label}.jsonl"
        write_jsonl(out, label_rows)
        repeat = repeat_multicar if label == TARGET_LABEL else repeat_debris if label == "抛洒物" else 1
        meta[label] = {
            "root": str(video_root.resolve()).replace("\\", "/"),
            "annotation": str(out.resolve()).replace("\\", "/"),
            "data_augment": False,
            "repeat_time": repeat,
            "length": len(label_rows),
        }
    meta_output.parent.mkdir(parents=True, exist_ok=True)
    meta_output.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build E9 strong-evidence 5-shot training and 10-shot diagnostic detection set.")
    p.add_argument("--review-csv", required=True)
    p.add_argument("--base-train-jsonl", required=True)
    p.add_argument("--full-test-jsonl")
    p.add_argument("--source-jsonl", action="append", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-count", type=int, default=5)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--video-root", required=True)
    p.add_argument("--repeat-multicar", type=int, default=8)
    p.add_argument("--repeat-debris", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    review = load_review(Path(args.review_csv))
    row_by_event, source_by_event = index_rows([Path(p) for p in args.source_jsonl])
    strong_ids = sorted([eid for eid, row in review.items() if is_strong_review(row)])
    found_strong_ids = [eid for eid in strong_ids if eid in row_by_event]
    missing = [eid for eid in strong_ids if eid not in row_by_event]
    train_ids, detect_ids = split_strong_ids(found_strong_ids, source_by_event, args.train_count, args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detect_rows = []
    for eid in detect_ids:
        row = inject_boundary(row_by_event[eid])
        row["strong_evidence_detect_id"] = eid
        detect_rows.append(row)

    full_test_rows = []
    if args.full_test_jsonl:
        full_test_events = set()
        for row in read_jsonl(Path(args.full_test_jsonl)):
            full_test_rows.append(inject_boundary(row))
            full_test_events.update(event_ids(str(row.get("video") or row.get("image") or "")))
        for eid in detect_ids:
            if eid not in full_test_events:
                full_test_rows.append(inject_boundary(row_by_event[eid]))

    train_rows = []
    detect_set = set(detect_ids)
    train_set = set(train_ids)
    for row in read_jsonl(Path(args.base_train_jsonl)):
        ids = set(event_ids(str(row.get("video") or row.get("image") or "")))
        if ids & detect_set:
            continue
        out = inject_boundary(row)
        if ids & train_set:
            eid = next(iter(ids & train_set))
            out = inject_strong_evidence(out, review[eid])
        train_rows.append(out)

    present_train_ids = set()
    for row in train_rows:
        present_train_ids.update(set(event_ids(str(row.get("video") or row.get("image") or ""))) & train_set)
    for eid in train_ids:
        if eid not in present_train_ids:
            out = inject_boundary(row_by_event[eid])
            out = inject_strong_evidence(out, review[eid])
            train_rows.append(out)

    train_jsonl = out_dir / "traffic_train_e9_strong5_boundary.jsonl"
    detect_jsonl = out_dir / "clear_collision_strong10_boundary.jsonl"
    full_test_jsonl = out_dir / "traffic_test_e9_strong5_boundary.jsonl"
    meta_json = out_dir / "holmesvau_data_e9_strong5_boundary_x8.json"
    by_class_dir = out_dir / "by_class_train_e9_strong5_boundary"
    manifest_csv = out_dir / "strong5_train_strong10_detect_manifest.csv"
    missing_txt = out_dir / "strong_collision_missing.txt"

    write_jsonl(train_jsonl, train_rows)
    write_jsonl(detect_jsonl, detect_rows)
    if args.full_test_jsonl:
        write_jsonl(full_test_jsonl, full_test_rows)

    manifest = []
    for role, ids in [("train_strong5", train_ids), ("detect_strong10", detect_ids)]:
        for eid in ids:
            row = row_by_event[eid]
            rv = review[eid]
            manifest.append(
                {
                    "role": role,
                    "event_id": eid,
                    "video": video_name(str(row.get("video") or row.get("image") or "")),
                    "source_jsonl": source_by_event.get(eid, ""),
                    "human_reason": rv.get("human_reason", ""),
                    "annotation_bucket": rv.get("annotation_bucket", ""),
                    "manual_note": rv.get("manual_note", ""),
                }
            )
    write_manifest(manifest_csv, manifest)
    missing_txt.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
    meta = build_meta(train_jsonl, meta_json, by_class_dir, Path(args.video_root), args.repeat_multicar, args.repeat_debris)

    print(
        json.dumps(
            {
                "strong_total_in_review": len(strong_ids),
                "found": len(found_strong_ids),
                "missing": missing,
                "train_ids": train_ids,
                "detect_ids": detect_ids,
                "train_jsonl": str(train_jsonl),
                "detect_jsonl": str(detect_jsonl),
                "full_test_jsonl": str(full_test_jsonl) if args.full_test_jsonl else "",
                "manifest_csv": str(manifest_csv),
                "meta_json": str(meta_json),
                "train_label_counts": dict(Counter(gt_label(r) for r in train_rows)),
                "detect_label_counts": dict(Counter(gt_label(r) for r in detect_rows)),
                "full_test_label_counts": dict(Counter(gt_label(r) for r in full_test_rows)) if args.full_test_jsonl else {},
                "meta": meta,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
