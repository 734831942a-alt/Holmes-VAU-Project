import argparse
import json
import random
from collections import Counter, defaultdict
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
DEFAULT_HARD_NEGATIVE_LABELS = ["拥堵", "异常停车", "二轮车辆闯入", "占道施工", "抛洒物"]


POSITIVE_PROMPT = """<video>
请判断这段视频是否属于“多车事故”。
判定标准：只有明确看到两辆及以上机动车发生碰撞、追尾、事故后共同滞留，或看到事故现场处置证据时，才回答“是”。
请用一句话先给出“是/否”，再说明关键依据。"""

NEGATIVE_PROMPT = """<video>
请判断这段视频是否属于“多车事故”。
判定标准：单车静止/占道优先属于“异常停车”；大范围排队低速优先属于“拥堵”；摩托车、电动车或非机动车进入主路优先属于“二轮车辆闯入”；施工锥桶、护栏、工程车导流优先属于“占道施工”；路面异物优先属于“抛洒物”。
请用一句话先给出“是/否”，再说明更合适的类别。"""


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


def video_name(row: dict[str, Any]) -> str:
    value = row.get("video") or row.get("image") or ""
    return Path(str(value).replace("\\", "/")).name


def label_from_video(name: str) -> str:
    name = Path(str(name).replace("\\", "/")).name
    prefix = name.split("_", 1)[0] if "_" in name else Path(name).stem
    return prefix if prefix in LABELS else ""


def extract_label(row: dict[str, Any]) -> str:
    for key in ("gt", "label", "category", "class", "cls"):
        value = row.get(key)
        if isinstance(value, str) and value in LABELS:
            return value
    return label_from_video(video_name(row))


def get_original_answer(row: dict[str, Any], max_chars: int) -> str:
    gt = row.get("gt")
    if isinstance(gt, str) and gt.strip():
        return gt.strip()[:max_chars]
    conversations = row.get("conversations")
    if isinstance(conversations, list):
        for turn in conversations:
            if isinstance(turn, dict) and turn.get("from") == "gpt":
                text = str(turn.get("value", "")).strip()
                if text:
                    return text[:max_chars]
    return ""


def clone_base_fields(row: dict[str, Any], sample_id: str, task: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("type", "video", "image"):
        if key in row:
            out[key] = row[key]
    out["id"] = sample_id
    out["task"] = task
    return out


def make_positive_aux(row: dict[str, Any], idx: int, max_context_chars: int) -> dict[str, Any]:
    original = get_original_answer(row, max_context_chars)
    answer = (
        "是，属于多车事故。判断依据应聚焦于两辆及以上机动车的碰撞、追尾、事故后共同滞留，"
        "或事故现场处置证据；若画面只有排队缓行或单车静止，则不能判为多车事故。"
    )
    if original:
        answer += f" 原始事件描述：{original}"
    out = clone_base_fields(row, f"{row.get('id', idx)}_multicar_pos_aux", "multicar_boundary_aux")
    out["conversations"] = [
        {"from": "human", "value": POSITIVE_PROMPT},
        {"from": "gpt", "value": answer},
    ]
    out["gt"] = TARGET_LABEL
    out["source_label"] = TARGET_LABEL
    out["e9_aux_type"] = "multicar_positive"
    return out


def make_negative_aux(row: dict[str, Any], label: str, idx: int, max_context_chars: int) -> dict[str, Any]:
    original = get_original_answer(row, max_context_chars)
    answer = (
        f"否，不属于多车事故；这段视频更符合“{label}”。"
        "不要仅因为车辆多、车辆慢行、单车停留或局部遮挡就判断为多车事故。"
    )
    if original:
        answer += f" 原始事件描述：{original}"
    out = clone_base_fields(row, f"{row.get('id', idx)}_multicar_neg_aux", "multicar_boundary_aux")
    out["conversations"] = [
        {"from": "human", "value": NEGATIVE_PROMPT},
        {"from": "gpt", "value": answer},
    ]
    out["gt"] = label
    out["source_label"] = label
    out["e9_aux_type"] = "multicar_hard_negative"
    return out


def pick_hard_negatives(
    rows_by_label: dict[str, list[tuple[int, dict[str, Any]]]],
    labels: list[str],
    total_needed: int,
    rng: random.Random,
) -> list[tuple[int, dict[str, Any], str]]:
    pools: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for label in labels:
        pool = list(rows_by_label.get(label, []))
        rng.shuffle(pool)
        pools[label] = pool

    selected: list[tuple[int, dict[str, Any], str]] = []
    cursor = 0
    active_labels = [label for label in labels if pools.get(label)]
    while len(selected) < total_needed and active_labels:
        label = active_labels[cursor % len(active_labels)]
        pool = pools[label]
        if pool:
            idx, row = pool.pop()
            selected.append((idx, row, label))
        active_labels = [x for x in active_labels if pools.get(x)]
        cursor += 1
    return selected


def build_augmented_train(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    input_path = Path(args.train_input)
    output_path = Path(args.train_output)
    rows = read_jsonl(input_path)

    rows_by_label: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    label_counts = Counter()
    unknown = 0
    for idx, row in enumerate(rows):
        label = extract_label(row)
        if label:
            rows_by_label[label].append((idx, row))
            label_counts[label] += 1
        else:
            unknown += 1

    output_rows = list(rows) if args.keep_original else []

    positive_rows = rows_by_label.get(TARGET_LABEL, [])
    for repeat_idx in range(args.positive_aux_repeat):
        for idx, row in positive_rows:
            aux = make_positive_aux(row, idx + repeat_idx * len(rows), args.max_context_chars)
            output_rows.append(aux)

    hard_labels = [x.strip() for x in args.hard_negative_labels.split(",") if x.strip()]
    total_negative_needed = len(positive_rows) * args.hard_negative_per_positive
    selected_negatives = pick_hard_negatives(rows_by_label, hard_labels, total_negative_needed, rng)
    for idx, row, label in selected_negatives:
        output_rows.append(make_negative_aux(row, label, idx, args.max_context_chars))

    if args.shuffle:
        rng.shuffle(output_rows)

    write_jsonl(output_path, output_rows)

    aux_counts = Counter(row.get("e9_aux_type", "original") for row in output_rows)
    output_label_counts = Counter(extract_label(row) for row in output_rows)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "input_total": len(rows),
        "output_total": len(output_rows),
        "input_label_counts": dict(label_counts),
        "output_label_counts": dict(output_label_counts),
        "unknown_input_label": unknown,
        "positive_aux": aux_counts.get("multicar_positive", 0),
        "hard_negative_aux": aux_counts.get("multicar_hard_negative", 0),
        "hard_negative_labels": hard_labels,
        "hard_negative_per_positive": args.hard_negative_per_positive,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build E9 training JSONL by appending multi-car accident boundary "
            "auxiliary samples. Test JSONL should stay original."
        )
    )
    parser.add_argument("--train-input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--keep-original", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--positive-aux-repeat", type=int, default=1)
    parser.add_argument("--hard-negative-per-positive", type=int, default=4)
    parser.add_argument(
        "--hard-negative-labels",
        default=",".join(DEFAULT_HARD_NEGATIVE_LABELS),
        help="Comma-separated labels sampled as no-multicar auxiliary examples.",
    )
    parser.add_argument("--max-context-chars", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_augmented_train(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
