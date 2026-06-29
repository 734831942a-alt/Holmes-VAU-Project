import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from simulate_sarp_v3 import parse_csv, simulate_sarp


DEFAULT_LABELS = "多车事故,拥堵,异常停车,占道施工,二轮车辆闯入,抛洒物"
MULTICAR_LABEL = "\u591a\u8f66\u4e8b\u6545"
CONGESTION_LABEL = "\u62e5\u5835"
SINGLE_STOP_LABEL = "\u5f02\u5e38\u505c\u8f66"
CONSTRUCTION_LABEL = "\u5360\u9053\u65bd\u5de5"
MOTORCYCLE_LABEL = "\u4e8c\u8f6e\u8f66\u8f86\u95ef\u5165"
DEBRIS_LABEL = "\u629b\u6d12\u7269"


def load_jsonl_map(path):
    data = {}
    if not path:
        return data
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            video = obj.get("video", "")
            if video:
                data[video] = obj
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


def video_prefix_label(obj, labels):
    video = obj.get("video", "")
    prefix = video.split("_")[0] if "_" in video else ""
    return prefix if prefix in labels else None


def field_gt_label(obj, labels):
    for key in ["gt_category", "label", "category", "gt_label"]:
        if key in obj:
            label = detect_label(obj.get(key), labels)
            if label:
                return label
    for key in ["gt", "answer", "target"]:
        if key in obj:
            label = detect_label(obj.get(key), labels)
            if label:
                return label
    for turn in obj.get("conversations", []) or []:
        if turn.get("from") == "gpt":
            label = detect_label(turn.get("value", ""), labels)
            if label:
                return label
    return None


def load_gt_labels(path, labels, source="auto"):
    gt_labels = {}
    totals = Counter()
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            video = obj.get("video", "")
            if source == "video":
                label = video_prefix_label(obj, labels)
            elif source == "fields":
                label = field_gt_label(obj, labels)
            else:
                label = field_gt_label(obj, labels) or video_prefix_label(obj, labels)
            label = label or "UNKNOWN"
            gt_labels[video] = label
            totals[label] += 1
    return gt_labels, totals


def parse_weight_map(text):
    weights = {}
    for item in parse_csv(text):
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        try:
            weights[key.strip()] = float(value.strip())
        except ValueError:
            continue
    return weights


def pick_pred_label(rec, labels, pred_field):
    for key in [pred_field, "pred_label", "prediction", "output", "response", "answer"]:
        if key in rec:
            label = detect_label(rec.get(key), labels)
            if label:
                return label
    return None


def parse_topk_labels(rec, labels, topk_field):
    keys = [topk_field, "topk", "top_k", "topk_labels", "pred_topk", "candidate_labels"]
    raw = None
    for key in keys:
        if key and key in rec:
            raw = rec.get(key)
            break
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


def get_ovd_payload(obj):
    if not isinstance(obj, dict):
        return {}
    ovd = obj.get("ovd") or obj.get("ovd_summary") or {}
    if isinstance(ovd, dict) and ovd:
        return ovd
    # Some summary files store module outputs directly at top level.
    if "sarp" in obj or "vscm" in obj or "vscm_debug" in obj:
        return obj
    return {}


def get_vscm_signal(ovd):
    vscm = ovd.get("vscm") or {}
    return bool(isinstance(vscm, dict) and vscm.get("triggered"))


def make_sarp_args(args):
    return SimpleNamespace(
        iou_thr=args.sarp_iou_thr,
        center_thr=args.sarp_center_thr,
        min_frames=args.sarp_min_frames,
        max_shift=args.sarp_max_shift,
        motorcycle_peak=args.motorcycle_peak,
        cone_peak=args.cone_peak,
        construction_vehicle_peak=args.construction_vehicle_peak,
        suppress_busy_scene=False,
        person_peak=args.person_peak,
        car_peak=35,
        truck_peak=8,
        local_suppression=args.sarp_local_suppression,
        local_iou_thr=args.sarp_local_iou_thr,
        local_center_scale=args.sarp_local_center_scale,
        local_min_hits=args.sarp_local_min_hits,
        local_suppress_labels=args.sarp_local_suppress_labels,
        risk_gated=args.sarp_risk_gated,
        hard_risk_gate=args.sarp_hard_risk_gate,
        strong_labels=args.sarp_strong_labels,
        strong_min_frames=args.sarp_strong_min_frames,
        strong_min_count=args.sarp_strong_min_count,
        strong_max_shift=args.sarp_strong_max_shift,
        strong_max_area_fold=args.sarp_strong_max_area_fold,
        weak_labels=args.sarp_weak_labels,
        allow_weak_trigger=args.sarp_allow_weak_trigger,
        weak_min_frames=args.sarp_weak_min_frames,
        weak_min_count=args.sarp_weak_min_count,
        weak_max_shift=args.sarp_weak_max_shift,
        weak_max_area_fold=args.sarp_weak_max_area_fold,
    )


def get_sarp_signal(ovd, args, sarp_args):
    sarp = ovd.get("sarp") or {}
    if not isinstance(sarp, dict) or not sarp.get("ok"):
        return False, {}
    if args.sarp_mode == "stored":
        return bool(sarp.get("triggered")), {"mode": "stored"}
    triggered, raw, reason, kept_tracks, stable_tracks, debug = simulate_sarp(sarp, sarp_args)
    debug = dict(debug)
    debug.update({
        "mode": "v3",
        "raw": raw,
        "reason": reason,
        "kept_track_count": len(kept_tracks),
        "stable_track_count": len(stable_tracks),
    })
    return triggered, debug


def sarp_conditioned_weight(base_label, ovd, args):
    if not args.conditioned_sarp:
        return args.sarp_weight

    weight_map = parse_weight_map(args.sarp_base_weights)
    weight = weight_map.get(base_label, args.sarp_other_weight)

    sarp = ovd.get("sarp") or {}
    semantic_peak = sarp.get("semantic_peak") if isinstance(sarp, dict) else {}
    semantic_peak = semantic_peak or {}
    if semantic_peak.get("traffic cone", 0) >= args.cone_peak or semantic_peak.get("construction vehicle", 0) >= args.construction_vehicle_peak:
        weight -= args.sarp_construction_penalty
    if semantic_peak.get("motorcycle", 0) >= args.motorcycle_peak:
        weight -= args.sarp_motorcycle_penalty
    return max(0.0, weight)


def topk_allows(label, topk_labels, require_topk, missing_policy):
    if not require_topk:
        return True
    if not topk_labels:
        return missing_policy == "allow"
    return label in topk_labels


def topk_rank(label, topk_labels):
    try:
        return topk_labels.index(label) + 1
    except ValueError:
        return None


def fuse_label(base_label, sarp_signal, vscm_signal, labels, args, ovd=None, topk_labels=None):
    if not base_label:
        return None, "no_base_label"
    ovd = ovd or {}
    topk_labels = topk_labels or []
    if args.topk_debris_rerank:
        rank = topk_rank(DEBRIS_LABEL, topk_labels)
        if (
            rank is not None
            and rank <= args.topk_debris_max_rank
            and base_label in args.topk_debris_rerank_bases
        ):
            return DEBRIS_LABEL, "topk_to_debris"

    scores = {label: 0.0 for label in labels}
    scores[base_label] = args.base_weight
    if sarp_signal:
        if topk_allows(DEBRIS_LABEL, topk_labels, args.require_debris_in_topk, args.topk_missing_policy):
            scores[DEBRIS_LABEL] += sarp_conditioned_weight(base_label, ovd, args)
    if vscm_signal:
        if topk_allows(MULTICAR_LABEL, topk_labels, args.require_multicar_in_topk, args.topk_missing_policy):
            scores[MULTICAR_LABEL] += args.vscm_weight

    best_label = base_label
    best_score = scores[base_label]
    for label, score in scores.items():
        if score > best_score + args.min_delta:
            best_label = label
            best_score = score

    if best_label != base_label:
        if best_label == DEBRIS_LABEL:
            return best_label, "sarp_to_debris"
        if best_label == MULTICAR_LABEL:
            return best_label, "vscm_to_multicar"
        return best_label, "score_override"
    if sarp_signal and base_label == DEBRIS_LABEL:
        return best_label, "sarp_confirm_debris"
    if vscm_signal and base_label == MULTICAR_LABEL:
        return best_label, "vscm_confirm_multicar"
    if sarp_signal and args.require_debris_in_topk and topk_labels and DEBRIS_LABEL not in topk_labels:
        return best_label, "sarp_blocked_by_topk"
    if vscm_signal and args.require_multicar_in_topk and topk_labels and MULTICAR_LABEL not in topk_labels:
        return best_label, "vscm_blocked_by_topk"
    if sarp_signal:
        return best_label, "sarp_insufficient_weight"
    return best_label, "unchanged"


def compute_metrics(rows, labels, pred_key):
    total = len(rows)
    correct = sum(1 for row in rows if row["gt"] == row.get(pred_key))
    per_class = {}
    matrix = {gt: {pd: 0 for pd in labels} for gt in labels}
    for gt in labels:
        tp = fp = fn = support = 0
        for row in rows:
            true_label = row["gt"]
            pred_label = row.get(pred_key)
            if true_label == gt:
                support += 1
            if true_label == gt and pred_label == gt:
                tp += 1
            elif true_label != gt and pred_label == gt:
                fp += 1
            elif true_label == gt and pred_label != gt:
                fn += 1
            if true_label in matrix and pred_label in matrix[true_label]:
                matrix[true_label][pred_label] += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[gt] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    macro_f1 = sum(v["f1"] for v in per_class.values()) / max(len(labels), 1)
    return {
        "accuracy": round(correct / total, 4) if total else 0.0,
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def print_signal_table(rows, labels):
    stats = defaultdict(Counter)
    for row in rows:
        gt = row["gt"]
        stats[gt]["total"] += 1
        if row["sarp_signal"]:
            stats[gt]["sarp"] += 1
        if row["vscm_signal"]:
            stats[gt]["vscm"] += 1
        if row["sarp_signal"] or row["vscm_signal"]:
            stats[gt]["any"] += 1
    print("类别               total   SARP   VSCM    any")
    print("-" * 54)
    for label in labels:
        c = stats[label]
        print(f"{label:<12} {c['total']:6d} {c['sarp']:6d} {c['vscm']:6d} {c['any']:6d}")


def print_sarp_breakdown(rows):
    debris_rows = [row for row in rows if row["gt"] == DEBRIS_LABEL]
    sarp_rows = [row for row in rows if row.get("sarp_signal")]
    sarp_tp = [row for row in sarp_rows if row["gt"] == DEBRIS_LABEL]
    sarp_fp = [row for row in sarp_rows if row["gt"] != DEBRIS_LABEL]
    debris_fn_by_sarp = [row for row in debris_rows if not row.get("sarp_signal")]
    debris_base_miss = [row for row in debris_rows if row.get("base_pred") and row.get("base_pred") != DEBRIS_LABEL]
    debris_topk_hit = [row for row in debris_rows if DEBRIS_LABEL in (row.get("topk_labels") or [])]

    precision = len(sarp_tp) / len(sarp_rows) if sarp_rows else 0.0
    recall = len(sarp_tp) / len(debris_rows) if debris_rows else 0.0
    print("\n=== SARP debris breakdown ===")
    print({
        "sarp_total": len(sarp_rows),
        "sarp_tp_debris": len(sarp_tp),
        "sarp_fp_non_debris": len(sarp_fp),
        "sarp_precision_as_debris": round(precision, 4),
        "sarp_recall_on_debris": round(recall, 4),
        "debris_support": len(debris_rows),
        "debris_without_sarp": len(debris_fn_by_sarp),
        "debris_base_miss": len(debris_base_miss),
        "debris_topk_contains_debris": len(debris_topk_hit),
    })

    print("SARP TP base_pred:", dict(Counter(row.get("base_pred") for row in sarp_tp)))
    print("SARP TP topk_has_debris:", dict(Counter(DEBRIS_LABEL in (row.get("topk_labels") or []) for row in sarp_tp)))
    print("SARP TP fusion_reason:", dict(Counter(row.get("fusion_reason") for row in sarp_tp)))
    print("SARP FP gt:", dict(Counter(row.get("gt") for row in sarp_fp)))
    print("SARP FP base_pred:", dict(Counter(row.get("base_pred") for row in sarp_fp)))
    print("SARP FP topk_has_debris:", dict(Counter(DEBRIS_LABEL in (row.get("topk_labels") or []) for row in sarp_fp)))
    print("SARP FP fusion_reason:", dict(Counter(row.get("fusion_reason") for row in sarp_fp)))
    print("Debris base_miss base_pred:", dict(Counter(row.get("base_pred") for row in debris_base_miss)))
    print("Debris base_miss SARP:", dict(Counter(bool(row.get("sarp_signal")) for row in debris_base_miss)))
    print("Debris base_miss topk_has_debris:", dict(Counter(DEBRIS_LABEL in (row.get("topk_labels") or []) for row in debris_base_miss)))


def print_topk_breakdown(rows):
    with_topk = [row for row in rows if row.get("topk_labels")]
    debris_rows = [row for row in rows if row["gt"] == DEBRIS_LABEL]
    non_debris_rows = [row for row in rows if row["gt"] != DEBRIS_LABEL]
    debris_in_topk = [row for row in rows if DEBRIS_LABEL in (row.get("topk_labels") or [])]
    debris_topk_tp = [row for row in debris_in_topk if row["gt"] == DEBRIS_LABEL]
    debris_topk_fp = [row for row in debris_in_topk if row["gt"] != DEBRIS_LABEL]
    debris_base_miss = [row for row in debris_rows if row.get("base_pred") and row.get("base_pred") != DEBRIS_LABEL]
    debris_base_miss_topk = [row for row in debris_base_miss if DEBRIS_LABEL in (row.get("topk_labels") or [])]

    precision = len(debris_topk_tp) / len(debris_in_topk) if debris_in_topk else 0.0
    recall = len(debris_topk_tp) / len(debris_rows) if debris_rows else 0.0
    print("\n=== top-k debris breakdown ===")
    print({
        "rows_with_topk": len(with_topk),
        "topk_contains_debris_total": len(debris_in_topk),
        "topk_debris_tp": len(debris_topk_tp),
        "topk_debris_fp": len(debris_topk_fp),
        "topk_debris_precision": round(precision, 4),
        "topk_debris_recall": round(recall, 4),
        "debris_base_miss": len(debris_base_miss),
        "debris_base_miss_topk_debris": len(debris_base_miss_topk),
    })
    print("topk_debris by gt:", dict(Counter(row.get("gt") for row in debris_in_topk)))
    print("topk_debris FP base_pred:", dict(Counter(row.get("base_pred") for row in debris_topk_fp)))
    print("topk_debris TP base_pred:", dict(Counter(row.get("base_pred") for row in debris_topk_tp)))
    print("topk_debris TP rank:", dict(Counter(topk_rank(DEBRIS_LABEL, row.get("topk_labels") or []) for row in debris_topk_tp)))
    print("topk_debris FP rank:", dict(Counter(topk_rank(DEBRIS_LABEL, row.get("topk_labels") or []) for row in debris_topk_fp)))
    print("debris base_miss topk labels:", dict(Counter(tuple(row.get("topk_labels") or []) for row in debris_base_miss_topk)))


def print_metrics(title, metrics, labels):
    print(f"\n=== {title} ===")
    print({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})
    for label in labels:
        print(label, metrics["per_class"][label])


def main():
    ap = argparse.ArgumentParser(description="Simulate class-specific evidence fusion for VSCM and SARP.")
    ap.add_argument("--gt", required=True, help="Ground-truth jsonl; label is inferred from video prefix.")
    ap.add_argument("--ovd", required=True, help="OVD summary jsonl containing vscm/sarp.")
    ap.add_argument("--pred", default="", help="Optional baseline prediction jsonl.")
    ap.add_argument("--pred-field", default="pred")
    ap.add_argument("--topk-field", default="topk_labels")
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument(
        "--gt-source",
        choices=["auto", "fields", "video"],
        default="video",
        help="How to extract GT labels. Default video matches the current traffic experiments' filename-prefix protocol.",
    )
    ap.add_argument("--examples", type=int, default=12)
    ap.add_argument("--base-weight", type=float, default=1.0)
    ap.add_argument("--sarp-weight", type=float, default=1.15)
    ap.add_argument("--vscm-weight", type=float, default=1.10)
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--sarp-mode", choices=["stored", "v3"], default="stored")
    ap.add_argument(
        "--conditioned-sarp",
        action="store_true",
        help="Use base-pred-conditioned SARP weights instead of one global SARP weight.",
    )
    ap.add_argument(
        "--sarp-base-weights",
        default=(
            "\u629b\u6d12\u7269:1.20,"
            "\u62e5\u5835:1.05,"
            "\u5f02\u5e38\u505c\u8f66:1.05,"
            "\u591a\u8f66\u4e8b\u6545:0.75,"
            "\u5360\u9053\u65bd\u5de5:0.35,"
            "\u4e8c\u8f6e\u8f66\u8f86\u95ef\u5165:0.35"
        ),
        help="Comma-separated base_label:weight map for SARP evidence.",
    )
    ap.add_argument("--sarp-other-weight", type=float, default=0.20)
    ap.add_argument("--sarp-construction-penalty", type=float, default=0.20)
    ap.add_argument("--sarp-motorcycle-penalty", type=float, default=0.15)
    ap.add_argument(
        "--require-debris-in-topk",
        action="store_true",
        help="Allow SARP to flip to debris only when debris appears in prediction top-k.",
    )
    ap.add_argument(
        "--require-multicar-in-topk",
        action="store_true",
        help="Allow VSCM to flip to multi-car accident only when multi-car accident appears in prediction top-k.",
    )
    ap.add_argument(
        "--topk-missing-policy",
        choices=["allow", "block"],
        default="allow",
        help="What to do when --require-*-in-topk is set but the prediction file has no top-k field.",
    )
    ap.add_argument(
        "--topk-debris-rerank",
        action="store_true",
        help="Use top-k itself as a debris reranker before SARP/VSCM score fusion.",
    )
    ap.add_argument(
        "--topk-debris-rerank-bases",
        default="\u62e5\u5835,\u5f02\u5e38\u505c\u8f66",
        type=parse_csv,
        help="Base predictions that may be reranked to debris when debris appears in top-k.",
    )
    ap.add_argument("--topk-debris-max-rank", type=int, default=1)

    ap.add_argument("--sarp-iou-thr", type=float, default=0.30)
    ap.add_argument("--sarp-center-thr", type=float, default=0.05)
    ap.add_argument("--sarp-min-frames", type=int, default=3)
    ap.add_argument("--sarp-max-shift", type=float, default=0.08)
    ap.add_argument("--sarp-local-suppression", action="store_true")
    ap.add_argument("--sarp-local-iou-thr", type=float, default=0.30)
    ap.add_argument("--sarp-local-center-scale", type=float, default=0.75)
    ap.add_argument("--sarp-local-min-hits", type=int, default=2)
    ap.add_argument(
        "--sarp-local-suppress-labels",
        default="car,truck,bus,motorcycle,person,traffic cone,construction vehicle,road barrier",
        type=parse_csv,
    )
    ap.add_argument("--sarp-risk-gated", action="store_true")
    ap.add_argument("--sarp-hard-risk-gate", action="store_true")
    ap.add_argument("--sarp-strong-labels", default="debris,road debris,trash,road spill", type=parse_csv)
    ap.add_argument("--sarp-strong-min-frames", type=int, default=3)
    ap.add_argument("--sarp-strong-min-count", type=int, default=3)
    ap.add_argument("--sarp-strong-max-shift", type=float, default=0.06)
    ap.add_argument("--sarp-strong-max-area-fold", type=float, default=5.0)
    ap.add_argument("--sarp-weak-labels", default="road obstacle,scattered object,foreign object,fallen object", type=parse_csv)
    ap.add_argument("--sarp-allow-weak-trigger", action="store_true")
    ap.add_argument("--sarp-weak-min-frames", type=int, default=4)
    ap.add_argument("--sarp-weak-min-count", type=int, default=4)
    ap.add_argument("--sarp-weak-max-shift", type=float, default=0.05)
    ap.add_argument("--sarp-weak-max-area-fold", type=float, default=4.0)

    ap.add_argument("--motorcycle-peak", type=int, default=2)
    ap.add_argument("--cone-peak", type=int, default=5)
    ap.add_argument("--construction-vehicle-peak", type=int, default=4)
    ap.add_argument("--person-peak", type=int, default=5)
    args = ap.parse_args()

    labels = parse_csv(args.labels)
    gt_labels, _ = load_gt_labels(args.gt, labels, source=args.gt_source)
    ovd_map = load_jsonl_map(args.ovd)
    pred_map = load_jsonl_map(args.pred) if args.pred else {}
    sarp_args = make_sarp_args(args)

    rows = []
    diag = Counter()
    for video, gt in gt_labels.items():
        if gt not in labels:
            continue
        diag["gt_target_total"] += 1
        ovd_obj = ovd_map.get(video, {})
        if ovd_obj:
            diag["ovd_matched"] += 1
        else:
            diag["ovd_missing"] += 1
        ovd = get_ovd_payload(ovd_obj)
        if ovd.get("sarp"):
            diag["sarp_present"] += 1
            if isinstance(ovd.get("sarp"), dict) and ovd["sarp"].get("ok"):
                diag["sarp_ok"] += 1
        if ovd.get("vscm"):
            diag["vscm_present"] += 1
        sarp_signal, sarp_debug = get_sarp_signal(ovd, args, sarp_args)
        vscm_signal = get_vscm_signal(ovd)
        row = {
            "video": video,
            "gt": gt,
            "sarp_signal": sarp_signal,
            "vscm_signal": vscm_signal,
            "sarp_debug": sarp_debug,
        }
        if pred_map:
            pred_obj = pred_map.get(video, {})
            if pred_obj:
                diag["pred_matched"] += 1
            else:
                diag["pred_missing"] += 1
            base = pick_pred_label(pred_obj, labels, args.pred_field)
            topk_labels = parse_topk_labels(pred_obj, labels, args.topk_field)
            if base:
                diag["base_label_found"] += 1
            else:
                diag["base_label_missing"] += 1
            if topk_labels:
                diag["topk_found"] += 1
            else:
                diag["topk_missing"] += 1
            sarp_effective_weight = sarp_conditioned_weight(base, ovd, args) if sarp_signal else 0.0
            fused, reason = fuse_label(base, sarp_signal, vscm_signal, labels, args, ovd=ovd, topk_labels=topk_labels)
            row.update({
                "base_pred": base,
                "topk_labels": topk_labels,
                "fused_pred": fused,
                "fusion_reason": reason,
                "sarp_effective_weight": round(sarp_effective_weight, 4),
            })
        rows.append(row)

    print_signal_table(rows, labels)
    print("\n=== diagnostics ===")
    print(dict(diag))
    if diag.get("gt_target_total", 0):
        if diag.get("ovd_matched", 0) == 0:
            print("WARNING: no OVD records matched GT videos; check --ovd path or video names.")
        if diag.get("sarp_present", 0) == 0 and diag.get("vscm_present", 0) == 0:
            print("WARNING: matched OVD records contain no SARP/VSCM modules; regenerate OVD summary with current ovd_video_summary.py.")
        if pred_map and diag.get("base_label_found", 0) == 0:
            print("WARNING: no baseline labels were parsed; check --pred path, --pred-field, or GT/pred video overlap.")

    if pred_map:
        base_metrics = compute_metrics(rows, labels, "base_pred")
        fused_metrics = compute_metrics(rows, labels, "fused_pred")
        print_metrics("baseline", base_metrics, labels)
        print_metrics("fused", fused_metrics, labels)
        print_sarp_breakdown(rows)
        print_topk_breakdown(rows)

        changes = Counter(row["fusion_reason"] for row in rows)
        print("\n=== fusion changes ===")
        print(dict(changes))

        print("\n=== changed examples ===")
        shown = 0
        for row in rows:
            if row.get("base_pred") != row.get("fused_pred"):
                print(json.dumps(row, ensure_ascii=False))
                shown += 1
                if shown >= args.examples:
                    break


if __name__ == "__main__":
    main()
