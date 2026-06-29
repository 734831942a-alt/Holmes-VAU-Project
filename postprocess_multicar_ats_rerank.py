import argparse
import json
import re
from collections import Counter
from pathlib import Path


DEFAULT_LABELS = (
    "\u591a\u8f66\u4e8b\u6545,"
    "\u62e5\u5835,"
    "\u5f02\u5e38\u505c\u8f66,"
    "\u5360\u9053\u65bd\u5de5,"
    "\u4e8c\u8f6e\u8f66\u8f86\u95ef\u5165,"
    "\u629b\u6d12\u7269"
)
TARGET_LABEL = "\u591a\u8f66\u4e8b\u6545"
DEFAULT_BASES = "\u62e5\u5835,\u5f02\u5e38\u505c\u8f66"


def parse_csv(text):
    return [x.strip() for x in (text or "").split(",") if x.strip()]


def load_jsonl_map(path):
    data = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            video = rec.get("video", "")
            if video:
                data[video] = rec
    return data


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
        r"\u4e8b\u6545\u7c7b\u578b\s*[:\uff1a]\s*['\"]?([^\uff0c\u3002\uff1b;:'\"\s]+)",
        r"\u7c7b\u522b\s*[:\uff1a]\s*['\"]?([^\uff0c\u3002\uff1b;:'\"\s]+)",
        r"\u7c7b\u578b\s*[:\uff1a]\s*['\"]?([^\uff0c\u3002\uff1b;:'\"\s]+)",
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
        parts = re.split(r"[,，|;；\s]+", raw)
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


def rank_of(label, labels):
    try:
        return labels.index(label) + 1
    except ValueError:
        return None


def should_rerank(base_label, aux_label, aux_topk, args):
    if base_label == args.target_label:
        return False, "already_target"
    if base_label not in args.rerank_bases:
        return False, "base_not_allowed"

    topk_rank = rank_of(args.target_label, aux_topk)
    aux_pred_hit = aux_label == args.target_label
    topk_hit = topk_rank is not None and topk_rank <= args.max_topk_rank

    if args.require_aux_pred and not aux_pred_hit:
        return False, "aux_pred_not_target"
    if args.require_topk and not topk_hit:
        return False, "topk_not_target"
    if not args.require_aux_pred and not args.require_topk and not (aux_pred_hit or topk_hit):
        return False, "no_aux_target_signal"

    return True, "ats_multicar_rerank"


def main():
    ap = argparse.ArgumentParser(
        description="Conservative multi-car rerank: keep E8 as base, use ATS/top-k only as accident verifier."
    )
    ap.add_argument("--base-pred-jsonl", required=True, help="Main non-ATS E8 prediction jsonl.")
    ap.add_argument("--aux-pred-jsonl", required=True, help="ATS prediction/top-k jsonl.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--base-pred-field", default="pred")
    ap.add_argument("--aux-pred-field", default="pred")
    ap.add_argument("--topk-field", default="topk_labels")
    ap.add_argument("--output-field", default="pred")
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument("--target-label", default=TARGET_LABEL)
    ap.add_argument("--rerank-bases", default=DEFAULT_BASES, type=parse_csv)
    ap.add_argument("--max-topk-rank", type=int, default=1)
    ap.add_argument("--require-aux-pred", action="store_true")
    ap.add_argument("--require-topk", action="store_true")
    ap.add_argument("--summary-json", default="")
    args = ap.parse_args()

    labels = parse_csv(args.labels)
    args.rerank_bases = set(args.rerank_bases)

    aux_map = load_jsonl_map(args.aux_pred_jsonl)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    changed = 0
    missing_aux = 0
    reason_counter = Counter()
    base_counter = Counter()
    aux_counter = Counter()
    examples = []

    with Path(args.base_pred_jsonl).open("r", encoding="utf-8") as f, out_path.open("w", encoding="utf-8") as w:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            total += 1
            video = rec.get("video", "")
            aux = aux_map.get(video, {})
            if not aux:
                missing_aux += 1

            original_pred = rec.get(args.base_pred_field)
            base_label = detect_label(original_pred, labels)
            aux_label = detect_label(aux.get(args.aux_pred_field), labels)
            aux_topk = parse_topk_labels(aux, labels, args.topk_field)

            base_counter[base_label or "NONE"] += 1
            aux_counter[aux_label or "NONE"] += 1

            rerank, reason = should_rerank(base_label, aux_label, aux_topk, args)
            reason_counter[reason] += 1

            rec["pred_before_multicar_ats_rerank"] = original_pred
            rec["multicar_ats_base_label"] = base_label
            rec["multicar_ats_aux_label"] = aux_label
            rec["multicar_ats_topk_labels"] = aux_topk
            rec["multicar_ats_rerank_applied"] = bool(rerank)
            rec["multicar_ats_rerank_reason"] = reason

            if rerank:
                changed += 1
                rec[args.output_field] = args.target_label
                if len(examples) < 30:
                    examples.append({
                        "video": video,
                        "base_label": base_label,
                        "aux_label": aux_label,
                        "aux_topk": aux_topk,
                        "old_pred": original_pred,
                        "new_pred": args.target_label,
                    })

            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "base_pred_jsonl": args.base_pred_jsonl,
        "aux_pred_jsonl": args.aux_pred_jsonl,
        "output": args.output,
        "total": total,
        "changed": changed,
        "missing_aux": missing_aux,
        "target_label": args.target_label,
        "rerank_bases": sorted(args.rerank_bases),
        "max_topk_rank": args.max_topk_rank,
        "require_aux_pred": args.require_aux_pred,
        "require_topk": args.require_topk,
        "reason_counter": dict(reason_counter),
        "base_counter": dict(base_counter),
        "aux_counter": dict(aux_counter),
        "examples": examples,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
