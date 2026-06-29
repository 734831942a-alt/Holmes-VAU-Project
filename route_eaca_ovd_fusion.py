import argparse
import json
import re
from collections import Counter
from pathlib import Path


MULTICAR_LABEL = "\u591a\u8f66\u4e8b\u6545"
CONGESTION_LABEL = "\u62e5\u5835"
PARKING_LABEL = "\u5f02\u5e38\u505c\u8f66"
CONSTRUCTION_LABEL = "\u5360\u9053\u65bd\u5de5"
TWOWHEEL_LABEL = "\u4e8c\u8f6e\u8f66\u8f86\u95ef\u5165"
DEBRIS_LABEL = "\u629b\u6d12\u7269"

DEFAULT_LABELS = [
    MULTICAR_LABEL,
    CONGESTION_LABEL,
    PARKING_LABEL,
    CONSTRUCTION_LABEL,
    TWOWHEEL_LABEL,
    DEBRIS_LABEL,
]


def parse_csv(text):
    if not text:
        return []
    return [x.strip() for x in str(text).split(",") if x.strip()]


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def by_video(rows):
    out = {}
    for row in rows:
        video = row.get("video")
        if video:
            out[video] = row
    return out


def label_from_video(video):
    return str(video or "").split("_")[0]


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
        r"\u4e8b\u6545\u7c7b\u578b\s*[:\uff1a]\s*['\"]?([^,\uff0c\u3002\uff1b;:'\"\s]+)",
        r"\u7c7b\u522b\s*[:\uff1a]\s*['\"]?([^,\uff0c\u3002\uff1b;:'\"\s]+)",
        r"\u7c7b\u578b\s*[:\uff1a]\s*['\"]?([^,\uff0c\u3002\uff1b;:'\"\s]+)",
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if not match:
            continue
        cand = match.group(1)
        for label in labels:
            if cand == label or label in cand:
                return label
    return hits[0] if hits else None


def pick_pred_text(row, field):
    for key in [field, "pred", "prediction", "output", "response", "answer"]:
        if key in row:
            return row.get(key) or ""
    return ""


def get_ovd_payload(row):
    if not isinstance(row, dict):
        return {}
    ovd = row.get("ovd") or row.get("ovd_summary") or {}
    if isinstance(ovd, dict) and ovd:
        return ovd
    if any(k in row for k in ["peak_count", "sum_count", "sarp", "vscm", "lcrm"]):
        return row
    return {}


def _lookup_count(container, keys):
    if not isinstance(container, dict):
        return 0
    best = 0
    for key in keys:
        try:
            best = max(best, int(container.get(key, 0) or 0))
        except (TypeError, ValueError):
            pass
    return best


def ovd_peak(ovd, keys):
    """Use peak-style counts only; sum_count is intentionally excluded to avoid broad busy-scene bias."""
    best = _lookup_count(ovd.get("peak_count"), keys)
    sarp = ovd.get("sarp") if isinstance(ovd, dict) else {}
    if isinstance(sarp, dict):
        best = max(best, _lookup_count(sarp.get("semantic_peak"), keys))
    return best


def eaca_multicar_signal(eaca, allowed_reasons, min_label_count):
    if not isinstance(eaca, dict):
        return False, "no_eaca"
    if eaca.get("eaca_label") != MULTICAR_LABEL:
        return False, "eaca_not_multicar"
    reason = str(eaca.get("eaca_reason") or "")
    if allowed_reasons and reason not in allowed_reasons:
        return False, "eaca_reason_blocked"
    label_count = int((eaca.get("clip_label_counts") or {}).get(MULTICAR_LABEL, 0) or 0)
    if min_label_count > 0 and label_count < min_label_count:
        return False, "eaca_label_count_low"
    return True, "eaca_multicar"


POSITIVE_COLLISION_WORDS = [
    "\u8ffd\u5c3e\u78b0\u649e",
    "\u53d1\u751f\u8ffd\u5c3e",
    "\u53d1\u751f\u78b0\u649e",
    "\u76f8\u649e",
    "\u78b0\u649e",
    "\u8ffd\u5c3e",
    "\u5250\u8e6d",
    "\u522e\u64e6",
    "\u8f66\u5934\u53d7\u635f",
    "\u660e\u663e\u53d7\u635f",
    "\u53d7\u635f",
    "\u4e8b\u6545\u73b0\u573a",
    "\u4e8b\u6545\u5904\u7f6e",
]

NEGATION_WORDS = [
    "\u672a\u89c1",
    "\u6ca1\u6709",
    "\u65e0",
    "\u672a\u53d1\u73b0",
    "\u672a\u51fa\u73b0",
    "\u4e0d\u5c5e\u4e8e",
    "\u975e\u4ea4\u901a\u4e8b\u6545",
]

SINGLE_STOP_WORDS = [
    "\u4e00\u8f86",
    "\u5355\u8f66",
    "\u5b8c\u5168\u9759\u6b62",
    "\u957f\u65f6\u95f4\u9759\u6b62",
    "\u505c\u9760",
    "\u6545\u969c\u505c\u8f66",
    "\u5f02\u5e38\u505c\u8f66",
]


def _norm_text(text):
    return "" if text is None else str(text)


def _has_negation_before(text, pos, window=10):
    prefix = text[max(0, pos - window):pos]
    return any(word in prefix for word in NEGATION_WORDS)


def eaca_collision_clip_count(eaca):
    count = 0
    for clip in (eaca or {}).get("clips", []) or []:
        raw = _norm_text(clip.get("raw"))
        found = False
        for word in POSITIVE_COLLISION_WORDS:
            start = 0
            while True:
                pos = raw.find(word, start)
                if pos < 0:
                    break
                if not _has_negation_before(raw, pos):
                    found = True
                    break
                start = pos + len(word)
            if found:
                break
        if found:
            count += 1
    return count


def eaca_single_stop_without_collision(eaca):
    raw_all = " ".join(_norm_text(clip.get("raw")) for clip in (eaca or {}).get("clips", []) or [])
    return any(word in raw_all for word in SINGLE_STOP_WORDS) and eaca_collision_clip_count(eaca) == 0


def route_label(base_label, ovd, eaca, args):
    motorcycle_peak = ovd_peak(ovd, ["motorcycle", "motorbike", "bicycle", "electric bike"])
    cone_peak = ovd_peak(ovd, ["traffic cone", "cone"])
    construction_vehicle_peak = ovd_peak(ovd, ["construction vehicle", "excavator", "crane"])
    barrier_peak = ovd_peak(ovd, ["road barrier", "barrier"])

    construction_strong = (
        cone_peak >= args.cone_peak
        or construction_vehicle_peak >= args.construction_vehicle_peak
        or (cone_peak >= args.cone_pair_peak and barrier_peak >= args.barrier_pair_peak)
    )
    eaca_ok, eaca_reason = eaca_multicar_signal(eaca, args.eaca_reasons, args.eaca_min_label_count)
    collision_clip_count = eaca_collision_clip_count(eaca)
    if eaca_ok and args.eaca_min_collision_clip_count > 0 and collision_clip_count < args.eaca_min_collision_clip_count:
        eaca_ok = False
        eaca_reason = "eaca_collision_clip_low"
    if eaca_ok and args.eaca_suppress_single_stop_without_collision and eaca_single_stop_without_collision(eaca):
        eaca_ok = False
        eaca_reason = "eaca_single_stop_blocked"

    evidence = {
        "motorcycle_peak": motorcycle_peak,
        "cone_peak": cone_peak,
        "construction_vehicle_peak": construction_vehicle_peak,
        "barrier_peak": barrier_peak,
        "eaca_gate": eaca_reason,
        "eaca_collision_clip_count": collision_clip_count,
        "construction_strong": construction_strong,
    }

    # Default behavior: OVD is a guardrail, not a classifier. It can veto an EACA
    # multicar overwrite, but it should not overwrite the E8 base label by itself.
    if args.ovd_action == "override":
        protected_bases = {CONGESTION_LABEL, PARKING_LABEL, MULTICAR_LABEL}
        if base_label in protected_bases and motorcycle_peak >= args.motorcycle_peak:
            return TWOWHEEL_LABEL, "ovd_twowheel", evidence
        if base_label in protected_bases and construction_strong:
            return CONSTRUCTION_LABEL, "ovd_construction", evidence

    if args.eaca_multicar and eaca_ok and base_label in {CONGESTION_LABEL, PARKING_LABEL}:
        if motorcycle_peak >= args.motorcycle_peak:
            return base_label, "ovd_veto_twowheel", evidence
        if construction_strong:
            return base_label, "ovd_veto_construction", evidence
        return MULTICAR_LABEL, eaca_reason, evidence

    return base_label, "base", evidence


def main():
    ap = argparse.ArgumentParser(
        description="Route-style fusion: EACA only verifies multicar; OVD only protects strong two-wheel/construction evidence."
    )
    ap.add_argument("--base-pred", required=True)
    ap.add_argument("--ovd", required=True)
    ap.add_argument("--eaca", default="")
    ap.add_argument("--output", required=True)
    ap.add_argument("--base-pred-field", default="pred")
    ap.add_argument("--fused-field", default="pred_route_fused")
    ap.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    ap.add_argument("--scope", choices=["base", "eaca"], default="base")
    ap.add_argument(
        "--ovd-action",
        choices=["protect", "override"],
        default="protect",
        help="protect: OVD only vetoes EACA multicar overrides; override: OVD may rewrite labels directly.",
    )
    ap.add_argument("--eaca-multicar", action="store_true")
    ap.add_argument("--eaca-reasons", default="repeated_accident_evidence", type=parse_csv)
    ap.add_argument("--eaca-min-label-count", type=int, default=0)
    ap.add_argument(
        "--eaca-min-collision-clip-count",
        type=int,
        default=0,
        help="Require non-negated collision evidence in at least this many EACA clips.",
    )
    ap.add_argument(
        "--eaca-suppress-single-stop-without-collision",
        action="store_true",
        help="Block EACA multicar override when clips describe only a single stopped vehicle and no collision evidence.",
    )
    ap.add_argument("--motorcycle-peak", type=int, default=3)
    ap.add_argument("--cone-peak", type=int, default=8)
    ap.add_argument("--construction-vehicle-peak", type=int, default=4)
    ap.add_argument("--cone-pair-peak", type=int, default=5)
    ap.add_argument("--barrier-pair-peak", type=int, default=3)
    args = ap.parse_args()

    labels = parse_csv(args.labels)
    base_rows = read_jsonl(args.base_pred)
    base_map = by_video(base_rows)
    ovd_map = by_video(read_jsonl(args.ovd))
    eaca_rows = read_jsonl(args.eaca) if args.eaca else []
    eaca_map = by_video(eaca_rows)

    if args.scope == "eaca":
        source_videos = [row.get("video") for row in eaca_rows if row.get("video")]
    else:
        source_videos = [row.get("video") for row in base_rows if row.get("video")]

    out = []
    stats = Counter()
    route_by_gt = Counter()
    changes_by_gt = Counter()
    route_detail_by_gt = Counter()

    for video in source_videos:
        base = dict(base_map.get(video, {}))
        if not base:
            continue
        gt = label_from_video(video)
        base_text = pick_pred_text(base, args.base_pred_field)
        base_label = detect_label(base_text, labels)
        ovd = get_ovd_payload(ovd_map.get(video, {}))
        eaca = eaca_map.get(video, {})

        fused_label, reason, evidence = route_label(base_label, ovd, eaca, args)
        fused_text = fused_label if fused_label and fused_label != base_label else base_text

        rec = dict(base)
        rec[args.fused_field] = fused_text
        rec["route_base_label"] = base_label
        rec["route_fused_label"] = fused_label
        rec["route_reason"] = reason
        rec["route_evidence"] = evidence
        if eaca:
            rec["route_eaca_label"] = eaca.get("eaca_label")
            rec["route_eaca_reason"] = eaca.get("eaca_reason")
            rec["route_eaca_signal_counts"] = eaca.get("signal_counts") or {}
            rec["route_eaca_clip_label_counts"] = eaca.get("clip_label_counts") or {}

        stats["total"] += 1
        if ovd:
            stats["ovd_matched"] += 1
        if eaca:
            stats["eaca_matched"] += 1
        if base_label:
            stats["base_label_found"] += 1
        else:
            stats["base_label_missing"] += 1
        route_by_gt[(gt, reason)] += 1
        route_detail_by_gt[(gt, base_label, fused_label, reason)] += 1
        if fused_label and fused_label != base_label:
            stats["changed"] += 1
            changes_by_gt[gt] += 1
        out.append(rec)

    write_jsonl(args.output, out)
    print("output:", args.output)
    print("stats:", dict(stats))
    print("changes_by_gt:", dict(changes_by_gt))
    print("route_by_gt:", {str(k): v for k, v in route_by_gt.items()})
    print("route_detail_by_gt:", {str(k): v for k, v in route_detail_by_gt.items()})


if __name__ == "__main__":
    main()
