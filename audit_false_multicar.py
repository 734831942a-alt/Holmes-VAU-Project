import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


LABELS = [
    "\u62e5\u5835",
    "\u5f02\u5e38\u505c\u8f66",
    "\u5360\u9053\u65bd\u5de5",
    "\u591a\u8f66\u4e8b\u6545",
    "\u4e8c\u8f6e\u8f66\u8f86\u95ef\u5165",
    "\u629b\u6d12\u7269",
]

ALIASES = {
    "\u591a\u8f66\u4e8b\u6545": [
        "\u4ea4\u901a\u4e8b\u6545",
        "\u591a\u8f66",
        "\u78b0\u649e",
        "\u8ffd\u5c3e",
    ],
    "\u4e8c\u8f6e\u8f66\u8f86\u95ef\u5165": [
        "\u4e8c\u8f6e\u95ef\u5165",
        "\u4e8c\u8f6e",
        "\u975e\u673a\u52a8\u8f66\u95ef\u5165",
        "\u6469\u6258\u8f66",
        "\u7535\u52a8\u8f66",
    ],
    "\u629b\u6d12\u7269": [
        "\u629b\u6492\u7269",
        "\u6563\u843d\u7269",
        "\u5f02\u7269",
        "\u8def\u9762\u969c\u788d",
        "\u969c\u788d\u7269",
    ],
}

TARGET = "\u591a\u8f66\u4e8b\u6545"


def norm(x):
    return "" if x is None else str(x).strip()


def flatten_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "\n".join(flatten_text(v) for v in x)
    if isinstance(x, dict):
        if "value" in x:
            return flatten_text(x.get("value"))
        if "content" in x:
            return flatten_text(x.get("content"))
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def key_to_label():
    m = {}
    for label in LABELS:
        m[label] = label
        for alias in ALIASES.get(label, []):
            m[alias] = label
    return m


def detect_label(text):
    text = norm(text)
    if not text:
        return None
    m = key_to_label()

    patterns = [
        r"(?:\u4e8b\u4ef6\u7c7b\u578b|\u4e8b\u6545\u7c7b\u578b|\u7c7b\u578b|\u7c7b\u522b)\s*[:：]\s*[\"'\u201c\u201d]?\s*([^，。；;：:\n\r]+)",
        r"(?:\u5224\u5b9a\u4e3a|\u5224\u65ad\u4e3a|\u8bc6\u522b\u4e3a|\u5c5e\u4e8e)\s*[\"'\u201c\u201d]?\s*([^，。；;：:\n\r]+)",
    ]
    hits = []
    for pat in patterns:
        for match in re.finditer(pat, text):
            span = match.group(1)
            label = detect_in_span(span, m)
            if label:
                hits.append((match.start(), label))
    if hits:
        hits.sort(key=lambda x: x[0])
        return hits[-1][1]

    best_pos = -1
    best_label = None
    for k, label in m.items():
        pos = text.rfind(k)
        if pos > best_pos:
            best_pos = pos
            best_label = label
    return best_label if best_pos >= 0 else None


def detect_in_span(span, mapping):
    span = norm(span)
    if span in mapping:
        return mapping[span]
    for k in sorted(mapping, key=len, reverse=True):
        if k in span:
            return mapping[k]
    return None


def get_label_from_video(rec):
    video = rec.get("video", "")
    if isinstance(video, str) and "_" in video:
        prefix = video.split("_")[0]
        if prefix in LABELS:
            return prefix
    return None


def get_gt_label(rec, gt_field, gt_source="filename"):
    if gt_source in ("filename", "auto"):
        label = get_label_from_video(rec)
        if label:
            return label
        if gt_source == "filename":
            return None

    for key in ["gt_category", "label", "category", "gt_label"]:
        if key in rec:
            label = detect_label(rec.get(key))
            if label:
                return label
    if gt_field in rec:
        label = detect_label(flatten_text(rec.get(gt_field)))
        if label:
            return label
    return None


def get_prompt(rec):
    if "prompt" in rec:
        return flatten_text(rec.get("prompt"))
    conv = rec.get("conversations")
    if isinstance(conv, list) and conv:
        return flatten_text(conv[0])
    return ""


def has_vscm_hint(text):
    text = norm(text)
    return (
        "\u5c40\u90e8\u8f66\u8f86\u9759\u6b62\u7ebf\u7d22" in text
        or "VSCM" in text
        or "\u8f66\u8f86\u9759\u6b62\u7ebf\u7d22" in text
    )


def get_ovd(rec, ovd_map):
    video = rec.get("video", "")
    if isinstance(rec.get("ovd_summary"), dict):
        return rec.get("ovd_summary")
    if isinstance(rec.get("ovd"), dict):
        return rec.get("ovd")
    obj = ovd_map.get(video)
    if isinstance(obj, dict):
        return obj.get("ovd") if isinstance(obj.get("ovd"), dict) else obj
    return {}


def load_ovd_map(path):
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    out = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            out[obj.get("video", "")] = obj
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-jsonl", required=True)
    ap.add_argument("--ovd-jsonl", default="")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--pred-field", default="pred")
    ap.add_argument("--gt-field", default="gt")
    ap.add_argument(
        "--gt-source",
        choices=["filename", "field", "auto"],
        default="filename",
        help="Use filename prefix as GT by default to keep audit aligned with traffic labels.",
    )
    ap.add_argument("--target-label", default=TARGET)
    args = ap.parse_args()

    ovd_map = load_ovd_map(args.ovd_jsonl)
    rows = []
    total = 0
    valid = 0
    pred_counter = Counter()
    false_to_target = Counter()
    target_recall_errors = Counter()
    hint_counter = Counter()

    with Path(args.pred_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            rec = json.loads(line)
            gt_label = get_gt_label(rec, args.gt_field, args.gt_source)
            pred_text = flatten_text(rec.get(args.pred_field))
            pred_label = detect_label(pred_text)
            if not gt_label or not pred_label:
                continue
            valid += 1
            pred_counter[pred_label] += 1

            prompt = get_prompt(rec)
            ovd = get_ovd(rec, ovd_map)
            vscm = ovd.get("vscm") if isinstance(ovd, dict) else None
            vscm_debug = ovd.get("vscm_debug") if isinstance(ovd, dict) else None
            if has_vscm_hint(prompt):
                hint_counter[gt_label] += 1

            is_false_to_target = gt_label != args.target_label and pred_label == args.target_label
            is_target_missed = gt_label == args.target_label and pred_label != args.target_label
            if not (is_false_to_target or is_target_missed):
                continue

            if is_false_to_target:
                false_to_target[gt_label] += 1
                error_type = "false_positive_to_target"
            else:
                target_recall_errors[pred_label] += 1
                error_type = "target_missed_as_other"

            row = {
                "error_type": error_type,
                "video": rec.get("video", ""),
                "gt_label": gt_label,
                "pred_label": pred_label,
                "has_vscm_hint": has_vscm_hint(prompt),
                "prompt_head": prompt[:500],
                "pred_head": pred_text[:500],
                "vscm_triggered": "",
                "vscm_static_count": "",
                "vscm_moving_count": "",
                "vscm_static_ratio": "",
                "vscm_proximity": "",
                "vscm_proximity_reason": "",
                "vscm_queue_like": "",
                "vscm_context_frames": "",
                "vscm_context_hits": "",
                "vscm_debug_reason": "",
                "ovd_motorcycle_peak": "",
                "ovd_vehicle_peak": "",
            }

            if isinstance(ovd, dict):
                peak = ovd.get("peak_count") or {}
                row["ovd_motorcycle_peak"] = peak.get("motorcycle", "")
                row["ovd_vehicle_peak"] = max(
                    [peak.get("car", 0), peak.get("truck", 0), peak.get("bus", 0)]
                )
            if isinstance(vscm, dict):
                row["vscm_triggered"] = vscm.get("triggered", "")
                row["vscm_static_count"] = vscm.get("static_count", "")
                row["vscm_moving_count"] = vscm.get("moving_count", "")
                row["vscm_static_ratio"] = vscm.get("static_ratio", "")
                row["vscm_proximity"] = vscm.get("proximity", "")
                row["vscm_proximity_reason"] = vscm.get("proximity_reason", "")
                row["vscm_queue_like"] = vscm.get("queue_like", "")
                row["vscm_context_frames"] = vscm.get("context_frames", "")
                row["vscm_context_hits"] = json.dumps(vscm.get("context_hits", {}), ensure_ascii=False)
            if isinstance(vscm_debug, dict):
                row["vscm_debug_reason"] = vscm_debug.get("none_reason", "")

            rows.append(row)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "error_type", "video", "gt_label", "pred_label", "has_vscm_hint", "prompt_head", "pred_head"
    ]
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("total_records =", total)
    print("valid_records =", valid)
    print("pred_counter =", dict(pred_counter))
    print("false_positive_to_target_by_gt =", dict(false_to_target))
    print("target_missed_as_other =", dict(target_recall_errors))
    print("vscm_hint_by_gt =", dict(hint_counter))
    print("saved =", out)


if __name__ == "__main__":
    main()
