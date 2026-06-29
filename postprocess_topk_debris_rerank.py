import argparse
import json
import re
from collections import Counter
from pathlib import Path


DEFAULT_LABELS = "多车事故,拥堵,异常停车,占道施工,二轮车辆闯入,抛洒物"
DEFAULT_BASES = "拥堵,异常停车"
TARGET_LABEL = "抛洒物"


def parse_csv(text):
    return [x.strip() for x in (text or "").split(",") if x.strip()]


def detect_label(text, labels):
    text = "" if text is None else str(text).strip()
    if not text:
        return None
    for label in labels:
        if text == label:
            return label
    hits = [label for label in labels if label in text]
    if len(hits) == 1:
        return hits[0]
    patterns = [
        r"事故类型\s*[:：]\s*['\"]?([^，。；;:'\"\s]+)",
        r"类别\s*[:：]\s*['\"]?([^，。；;:'\"\s]+)",
        r"类型\s*[:：]\s*['\"]?([^，。；;:'\"\s]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            cand = m.group(1)
            for label in labels:
                if cand == label or label in cand:
                    return label
    return hits[0] if hits else None


def parse_topk_labels(rec, labels, topk_field):
    raw = rec.get(topk_field)
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = re.split(r"[,，/|;；\s]+", raw)
    elif isinstance(raw, list):
        parts = raw
    else:
        parts = [raw]

    out = []
    for item in parts:
        if isinstance(item, dict):
            text = item.get("label") or item.get("class") or item.get("name") or item.get("pred")
        else:
            text = item
        label = detect_label(text, labels)
        if label and label not in out:
            out.append(label)
    return out


def should_rerank(base_label, topk_labels, bases, target_label, max_rank):
    if base_label not in bases:
        return False
    if not topk_labels:
        return False
    try:
        rank = topk_labels.index(target_label) + 1
    except ValueError:
        return False
    return rank <= max_rank


def main():
    ap = argparse.ArgumentParser(description="Conservative top-k rerank: base congestion/parking + top1 debris -> debris.")
    ap.add_argument("--input", required=True, help="Prediction jsonl with pred and topk_labels fields.")
    ap.add_argument("--output", required=True, help="Output postprocessed jsonl.")
    ap.add_argument("--pred-field", default="pred")
    ap.add_argument("--topk-field", default="topk_labels")
    ap.add_argument("--output-field", default="pred", help="Field to write final prediction label/text into.")
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument("--target-label", default=TARGET_LABEL)
    ap.add_argument("--rerank-bases", default=DEFAULT_BASES)
    ap.add_argument("--max-rank", type=int, default=1)
    ap.add_argument("--summary-json", default="")
    args = ap.parse_args()

    labels = parse_csv(args.labels)
    bases = set(parse_csv(args.rerank_bases))

    total = 0
    changed = 0
    base_counter = Counter()
    topk_counter = Counter()
    reason_counter = Counter()
    examples = []

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Path(args.input).open("r", encoding="utf-8") as f, out_path.open("w", encoding="utf-8") as w:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            total += 1

            original_pred = rec.get(args.pred_field)
            base_label = detect_label(original_pred, labels)
            topk_labels = parse_topk_labels(rec, labels, args.topk_field)
            base_counter[base_label or "NONE"] += 1
            topk_counter[tuple(topk_labels)] += 1

            reranked = should_rerank(base_label, topk_labels, bases, args.target_label, args.max_rank)
            rec["pred_before_topk_rerank"] = original_pred
            rec["topk_rerank_base_label"] = base_label
            rec["topk_rerank_labels"] = topk_labels
            rec["topk_rerank_applied"] = bool(reranked)

            if reranked:
                changed += 1
                reason = f"base_{base_label}_top{args.max_rank}_{args.target_label}"
                reason_counter[reason] += 1
                rec["topk_rerank_reason"] = reason
                rec[args.output_field] = args.target_label
                if len(examples) < 20:
                    examples.append({
                        "video": rec.get("video", ""),
                        "base_label": base_label,
                        "topk_labels": topk_labels,
                        "old_pred": original_pred,
                        "new_pred": args.target_label,
                    })
            else:
                rec["topk_rerank_reason"] = ""

            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "input": args.input,
        "output": args.output,
        "total": total,
        "changed": changed,
        "target_label": args.target_label,
        "rerank_bases": sorted(bases),
        "max_rank": args.max_rank,
        "reason_counter": dict(reason_counter),
        "base_counter": dict(base_counter),
        "examples": examples,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
