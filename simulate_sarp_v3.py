import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_TARGETS = "多车事故,拥堵,异常停车,占道施工,二轮车辆闯入,抛洒物"


def parse_csv(text):
    return [x.strip() for x in (text or "").split(",") if x.strip()]


def load_labels(path):
    labels = {}
    totals = Counter()
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            video = obj.get("video", "")
            label = video.split("_")[0] if "_" in video else "UNKNOWN"
            labels[video] = label
            totals[label] += 1
    return labels, totals


def center_distance(a, b):
    ax, ay = a.get("center") or [0.0, 0.0]
    bx, by = b.get("center") or [0.0, 0.0]
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def box_iou(a, b):
    ab = a.get("box") or [0, 0, 0, 0]
    bb = b.get("box") or [0, 0, 0, 0]
    ix1 = max(ab[0], bb[0])
    iy1 = max(ab[1], bb[1])
    ix2 = min(ab[2], bb[2])
    iy2 = min(ab[3], bb[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ab[2] - ab[0]) * max(0.0, ab[3] - ab[1])
    area_b = max(0.0, bb[2] - bb[0]) * max(0.0, bb[3] - bb[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def raw_box_iou(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def box_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def suppress_track_locally(track, negatives_by_frame, args):
    checked = 0
    hits = Counter()
    for item in track:
        frame = item.get("frame")
        candidate_box = item.get("box") or [0, 0, 0, 0]
        candidate_center = box_center(candidate_box)
        for neg in negatives_by_frame.get(frame, []):
            neg_label = neg.get("label", "")
            if neg_label not in args.local_suppress_labels:
                continue
            neg_box = neg.get("box") or [0, 0, 0, 0]
            iou = raw_box_iou(candidate_box, neg_box)
            neg_center = box_center(neg_box)
            neg_w = max(1.0, neg_box[2] - neg_box[0])
            neg_h = max(1.0, neg_box[3] - neg_box[1])
            near = (
                abs(candidate_center[0] - neg_center[0]) <= args.local_center_scale * neg_w
                and abs(candidate_center[1] - neg_center[1]) <= args.local_center_scale * neg_h
            )
            if iou >= args.local_iou_thr or near:
                checked += 1
                hits[neg_label] += 1
                break
    if checked >= args.local_min_hits:
        reason = "local_" + hits.most_common(1)[0][0].replace(" ", "_")
        return True, reason
    return False, None


def track_has_strong_label(track, strong_labels):
    return any(label in strong_labels for label in track.get("labels", {}))


def track_area_fold(track):
    areas = [
        float(item.get("area_ratio", 0.0))
        for item in track.get("track", [])
        if float(item.get("area_ratio", 0.0)) > 0
    ]
    if not areas:
        return 1.0
    return max(areas) / max(min(areas), 1e-9)


def track_passes_weak_gate(track, args):
    labels = set(track.get("labels", {}))
    if not labels.intersection(args.weak_labels):
        return False
    return (
        track.get("frames", 0) >= args.weak_min_frames
        and track.get("count", 0) >= args.weak_min_count
        and track.get("max_shift", 1.0) <= args.weak_max_shift
        and track_area_fold(track) <= args.weak_max_area_fold
    )


def track_passes_strong_gate(track, args):
    return (
        track.get("frames", 0) >= args.strong_min_frames
        and track.get("count", 0) >= args.strong_min_count
        and track.get("max_shift", 1.0) <= args.strong_max_shift
        and track_area_fold(track) <= args.strong_max_area_fold
    )


def high_risk_scene(semantic_peak, args):
    return (
        semantic_peak.get("motorcycle", 0) >= args.motorcycle_peak
        or semantic_peak.get("traffic cone", 0) >= args.cone_peak
        or semantic_peak.get("construction vehicle", 0) >= args.construction_vehicle_peak
        or semantic_peak.get("person", 0) >= args.person_peak
    )


def build_tracks(candidates, iou_thr=0.30, center_thr=0.05):
    tracks = []
    for item in sorted(candidates, key=lambda x: (x.get("frame", 0), x.get("label", ""))):
        best_idx = None
        best_score = -1.0
        for idx, track in enumerate(tracks):
            last = track[-1]
            if item.get("frame") == last.get("frame"):
                continue
            iou = box_iou(item, last)
            dist = center_distance(item, last)
            if iou >= iou_thr or dist <= center_thr:
                score = iou - dist
                if score > best_score:
                    best_score = score
                    best_idx = idx
        if best_idx is None:
            tracks.append([item])
        else:
            tracks[best_idx].append(item)
    return tracks


def simulate_sarp(sarp, args):
    candidates = sarp.get("candidates")
    if candidates is None:
        candidates = sarp.get("examples") or []

    tracks = build_tracks(candidates, iou_thr=args.iou_thr, center_thr=args.center_thr)
    stable_tracks = []
    for track in tracks:
        frames = {x.get("frame") for x in track}
        if len(frames) < args.min_frames:
            continue
        centers = [x.get("center") or [0.0, 0.0] for x in track]
        max_shift = 0.0
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                shift = ((centers[i][0] - centers[j][0]) ** 2 + (centers[i][1] - centers[j][1]) ** 2) ** 0.5
                max_shift = max(max_shift, shift)
        if max_shift > args.max_shift:
            continue
        labels = Counter(x.get("label") for x in track)
        stable_tracks.append({
            "frames": len(frames),
            "count": len(track),
            "labels": dict(labels),
            "max_shift": round(max_shift, 4),
            "track": track,
            "examples": track[:5],
        })

    semantic_peak = sarp.get("semantic_peak") or {}
    strong_labels = set(args.strong_labels)
    risk_scene = high_risk_scene(semantic_peak, args) if args.risk_gated else False
    risk_removed = 0
    risk_strong_removed = 0
    risk_weak_support = 0
    risk_weak_triggered = 0
    gated_tracks = stable_tracks
    if risk_scene:
        gated_tracks = []
        for track in stable_tracks:
            if track_has_strong_label(track, strong_labels):
                if track_passes_strong_gate(track, args):
                    gated_tracks.append(track)
                else:
                    risk_strong_removed += 1
            elif (not args.hard_risk_gate) and track_passes_weak_gate(track, args):
                risk_weak_support += 1
                if args.allow_weak_trigger:
                    risk_weak_triggered += 1
                    gated_tracks.append(track)
        risk_removed = len(stable_tracks) - len(gated_tracks)

    negatives_by_frame = {}
    for frame_rec in sarp.get("negative_boxes_by_frame") or []:
        negatives_by_frame[frame_rec.get("frame")] = frame_rec.get("boxes") or []

    kept_tracks = gated_tracks
    local_reasons = Counter()
    if args.local_suppression and negatives_by_frame:
        kept_tracks = []
        for track in gated_tracks:
            suppressed, reason = suppress_track_locally(track.get("track", []), negatives_by_frame, args)
            if suppressed:
                local_reasons[reason or "local_negative"] += 1
            else:
                kept_tracks.append(track)

    triggered_before_suppression = bool(kept_tracks)
    suppression_reason = None
    if args.local_suppression:
        if stable_tracks and not gated_tracks and risk_scene:
            suppression_reason = "risk_gate_weak_labels"
        elif gated_tracks and not kept_tracks and local_reasons:
            suppression_reason = local_reasons.most_common(1)[0][0]
    elif triggered_before_suppression:
        if semantic_peak.get("motorcycle", 0) >= args.motorcycle_peak:
            suppression_reason = "motorcycle_evidence"
        elif semantic_peak.get("traffic cone", 0) >= args.cone_peak or semantic_peak.get("construction vehicle", 0) >= args.construction_vehicle_peak:
            suppression_reason = "construction_evidence"
        elif args.suppress_busy_scene and (
            semantic_peak.get("person", 0) >= args.person_peak
            or semantic_peak.get("car", 0) >= args.car_peak
            or semantic_peak.get("truck", 0) >= args.truck_peak
        ):
            suppression_reason = "busy_scene_evidence"

    triggered = triggered_before_suppression and suppression_reason is None
    debug = {
        "risk_gated": args.risk_gated,
        "high_risk_scene": risk_scene,
        "risk_gate_removed": risk_removed,
        "risk_strong_removed": risk_strong_removed,
        "risk_weak_support": risk_weak_support,
        "risk_weak_triggered": risk_weak_triggered,
        "gated_track_count": len(gated_tracks),
        "strong_labels": sorted(strong_labels),
        "weak_labels": sorted(args.weak_labels),
    }
    return triggered, bool(stable_tracks), suppression_reason, kept_tracks, stable_tracks, debug


def main():
    ap = argparse.ArgumentParser(description="Offline SARP v3 stable-candidate simulation.")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--ovd", required=True)
    ap.add_argument("--targets", default=DEFAULT_TARGETS)
    ap.add_argument("--examples", type=int, default=8)
    ap.add_argument("--iou-thr", type=float, default=0.30)
    ap.add_argument("--center-thr", type=float, default=0.05)
    ap.add_argument("--min-frames", type=int, default=3)
    ap.add_argument("--max-shift", type=float, default=0.08)
    ap.add_argument("--motorcycle-peak", type=int, default=2)
    ap.add_argument("--cone-peak", type=int, default=5)
    ap.add_argument("--construction-vehicle-peak", type=int, default=4)
    ap.add_argument("--suppress-busy-scene", action="store_true")
    ap.add_argument("--person-peak", type=int, default=5)
    ap.add_argument("--car-peak", type=int, default=35)
    ap.add_argument("--truck-peak", type=int, default=8)
    ap.add_argument("--local-suppression", action="store_true", help="Suppress stable tracks only when nearby negative boxes explain them.")
    ap.add_argument("--local-iou-thr", type=float, default=0.30)
    ap.add_argument("--local-center-scale", type=float, default=0.75)
    ap.add_argument("--local-min-hits", type=int, default=2)
    ap.add_argument(
        "--local-suppress-labels",
        default="car,truck,bus,motorcycle,person,traffic cone,construction vehicle,road barrier",
        type=parse_csv,
    )
    ap.add_argument(
        "--risk-gated",
        action="store_true",
        help="In motorcycle/construction/person-heavy scenes, only keep stable tracks with stronger debris-like labels.",
    )
    ap.add_argument(
        "--hard-risk-gate",
        action="store_true",
        help="Use the old hard risk gate: high-risk scenes only allow strong labels, no weak-label recovery.",
    )
    ap.add_argument(
        "--strong-labels",
        default="debris,road debris,trash,road spill",
        type=parse_csv,
        help="Stable labels allowed through the risk gate in high-risk scenes.",
    )
    ap.add_argument("--strong-min-frames", type=int, default=3)
    ap.add_argument("--strong-min-count", type=int, default=3)
    ap.add_argument("--strong-max-shift", type=float, default=0.06)
    ap.add_argument("--strong-max-area-fold", type=float, default=5.0)
    ap.add_argument(
        "--weak-labels",
        default="road obstacle,scattered object,foreign object,fallen object",
        type=parse_csv,
        help="Weak SARP labels that may pass a high-risk scene only when the track is very stable.",
    )
    ap.add_argument(
        "--allow-weak-trigger",
        action="store_true",
        help="Allow weak labels to trigger after passing the weak stability gate. By default weak labels are support-only.",
    )
    ap.add_argument("--weak-min-frames", type=int, default=4)
    ap.add_argument("--weak-min-count", type=int, default=4)
    ap.add_argument("--weak-max-shift", type=float, default=0.05)
    ap.add_argument("--weak-max-area-fold", type=float, default=4.0)
    args = ap.parse_args()

    targets = parse_csv(args.targets)
    labels, totals = load_labels(args.gt)
    stats = defaultdict(Counter)
    reasons = defaultdict(Counter)
    examples = defaultdict(list)

    with Path(args.ovd).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            video = obj.get("video", "")
            label = labels.get(video, video.split("_")[0] if "_" in video else "UNKNOWN")
            if targets and label not in targets:
                continue

            sarp = ((obj.get("ovd") or {}).get("sarp") or {})
            if not sarp.get("ok"):
                stats[label]["not_ok"] += 1
                continue
            stats[label]["ok"] += 1
            triggered, raw, reason, kept_tracks, stable_tracks, debug = simulate_sarp(sarp, args)
            if raw:
                stats[label]["raw_triggered"] += 1
            if debug.get("high_risk_scene"):
                stats[label]["high_risk_scene"] += 1
            if debug.get("risk_gate_removed"):
                stats[label]["risk_gate_removed"] += debug["risk_gate_removed"]
            if debug.get("risk_strong_removed"):
                stats[label]["risk_strong_removed"] += debug["risk_strong_removed"]
            if debug.get("risk_weak_support"):
                stats[label]["risk_weak_support"] += debug["risk_weak_support"]
            if debug.get("risk_weak_triggered"):
                stats[label]["risk_weak_triggered"] += debug["risk_weak_triggered"]
            if triggered:
                stats[label]["triggered"] += 1
            else:
                stats[label]["none"] += 1
            if reason:
                reasons[label][reason] += 1
            if raw and len(examples[label]) < args.examples:
                examples[label].append({
                    "video": video,
                    "triggered": triggered,
                    "raw_triggered": raw,
                    "suppression_reason": reason,
                    "stable_track_count": len(stable_tracks),
                    "kept_track_count": len(kept_tracks),
                    "risk_debug": debug,
                    "semantic_peak": sarp.get("semantic_peak"),
                    "label_hits": sarp.get("label_hits"),
                    "tracks": kept_tracks[:3],
                })

    print("类别               总数   SARP_OK  stable_raw  triggered  none")
    print("-" * 72)
    for label in targets:
        c = stats[label]
        total = totals.get(label, 0)
        ok = c.get("ok", 0)
        print(
            f"{label:<12} {total:6d} {ok:8d} "
            f"{c.get('raw_triggered', 0):10d} "
            f"{c.get('triggered', 0):10d} "
            f"{c.get('none', 0):5d}"
        )

    print("\n=== rates within SARP_OK ===")
    for label in targets:
        ok = max(stats[label].get("ok", 0), 1)
        print(label, {
            "stable_raw": round(stats[label].get("raw_triggered", 0) / ok, 3),
            "high_risk": round(stats[label].get("high_risk_scene", 0) / ok, 3),
            "weak_support": round(stats[label].get("risk_weak_support", 0) / ok, 3),
            "weak_trigger": round(stats[label].get("risk_weak_triggered", 0) / ok, 3),
            "triggered": round(stats[label].get("triggered", 0) / ok, 3),
        })

    print("\n=== suppression counts ===")
    for label in targets:
        if reasons[label]:
            print(label, dict(reasons[label]))

    print("\n=== examples ===")
    for label in targets:
        if examples[label]:
            print("\n", label)
            for item in examples[label]:
                print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
