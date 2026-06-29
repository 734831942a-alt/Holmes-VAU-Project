import argparse
import csv
import json
from collections import Counter
from pathlib import Path


TARGET = "\u591a\u8f66\u4e8b\u6545"


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


def load_jsonl_map(path):
    out = {}
    if not path:
        return out
    p = Path(path)
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            video = obj.get("video", "")
            if video:
                out[video] = obj
    return out


def get_prompt(rec):
    if not isinstance(rec, dict):
        return ""
    if "prompt" in rec:
        return flatten_text(rec.get("prompt"))
    conv = rec.get("conversations")
    if isinstance(conv, list) and conv:
        return flatten_text(conv[0])
    return ""


def has_vscm_hint(text):
    text = "" if text is None else str(text)
    return (
        "\u5c40\u90e8\u8f66\u8f86\u9759\u6b62\u7ebf\u7d22" in text
        or "\u8f66\u8f86\u9759\u6b62\u7ebf\u7d22" in text
        or "VSCM" in text
    )


def pick_ovd(ovd_map, pred_rec):
    if isinstance(pred_rec, dict):
        if isinstance(pred_rec.get("ovd_summary"), dict):
            return pred_rec.get("ovd_summary")
        if isinstance(pred_rec.get("ovd"), dict):
            return pred_rec.get("ovd")
    obj = ovd_map.get(pred_rec.get("video", "") if isinstance(pred_rec, dict) else "")
    if isinstance(obj, dict):
        return obj.get("ovd") if isinstance(obj.get("ovd"), dict) else obj
    return {}


def main():
    ap = argparse.ArgumentParser(
        description="Audit multi-car errors using eval_traffic_abnormal_only.py error CSV as source of truth."
    )
    ap.add_argument("--error-csv", required=True, help="CSV produced by eval_traffic_abnormal_only.py --error-csv")
    ap.add_argument("--pred-jsonl", required=True, help="Prediction JSONL used by the same eval run")
    ap.add_argument("--ovd-jsonl", default="", help="Optional OVD summary JSONL")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--target-label", default=TARGET)
    args = ap.parse_args()

    pred_map = load_jsonl_map(args.pred_jsonl)
    ovd_map = load_jsonl_map(args.ovd_jsonl)

    rows = []
    with Path(args.error_csv).open("r", encoding="utf-8-sig", newline="") as f:
        for err in csv.DictReader(f):
            gt_label = err.get("gt_label", "")
            pred_label = err.get("pred_label", "")
            if gt_label != args.target_label and pred_label != args.target_label:
                continue

            video = err.get("video", "")
            pred_rec = pred_map.get(video, {})
            prompt = get_prompt(pred_rec)
            ovd = pick_ovd(ovd_map, pred_rec)
            vscm = ovd.get("vscm") if isinstance(ovd, dict) else {}
            if not isinstance(vscm, dict):
                vscm = {}
            peak = ovd.get("peak_count") if isinstance(ovd, dict) else {}
            if not isinstance(peak, dict):
                peak = {}

            if pred_label == args.target_label and gt_label != args.target_label:
                error_type = "false_positive_to_target"
            else:
                error_type = "target_missed_as_other"

            rows.append(
                {
                    "error_type": error_type,
                    "index": err.get("index", ""),
                    "video": video,
                    "gt_label": gt_label,
                    "pred_label": pred_label,
                    "has_vscm_hint": has_vscm_hint(prompt),
                    "prompt_head": prompt[:500],
                    "pred_head": err.get("pred_text", "")[:500],
                    "vscm_triggered": vscm.get("triggered", ""),
                    "vscm_static_count": vscm.get("static_count", ""),
                    "vscm_moving_count": vscm.get("moving_count", ""),
                    "vscm_static_ratio": vscm.get("static_ratio", ""),
                    "vscm_proximity": vscm.get("proximity", ""),
                    "vscm_proximity_reason": vscm.get("proximity_reason", ""),
                    "vscm_queue_like": vscm.get("queue_like", ""),
                    "vscm_context_frames": vscm.get("context_frames", ""),
                    "vscm_context_hits": json.dumps(vscm.get("context_hits", {}), ensure_ascii=False),
                    "ovd_motorcycle_peak": peak.get("motorcycle", ""),
                    "ovd_vehicle_peak": max([peak.get("car", 0), peak.get("truck", 0), peak.get("bus", 0)]),
                }
            )

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "error_type",
        "index",
        "video",
        "gt_label",
        "pred_label",
        "has_vscm_hint",
    ]
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("rows =", len(rows))
    print("error_type =", dict(Counter(r["error_type"] for r in rows)))
    print("gt_pred =", dict(Counter((r["gt_label"], r["pred_label"]) for r in rows)))
    print("has_vscm_hint =", dict(Counter(str(r["has_vscm_hint"]) for r in rows)))
    print("saved =", out)


if __name__ == "__main__":
    main()
