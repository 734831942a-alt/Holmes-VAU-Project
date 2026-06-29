import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_TARGETS = "\u591a\u8f66\u4e8b\u6545,\u62e5\u5835,\u5f02\u5e38\u505c\u8f66,\u5360\u9053\u65bd\u5de5,\u4e8c\u8f6e\u8f66\u8f86\u95ef\u5165,\u629b\u6d12\u7269"


def parse_csv(x):
    return [v.strip() for v in (x or "").split(",") if v.strip()]


def load_gt(path):
    labels = {}
    totals = Counter()
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            video = obj.get("video", "")
            label = video.split("_")[0] if video else ""
            labels[video] = label
            totals[label] += 1
    return labels, totals


def get_peak(ovd, key):
    peak = ovd.get("peak_count") or {}
    return peak.get(key, 0) if isinstance(peak, dict) else 0


def ctx(vscm, key):
    ch = vscm.get("context_hits") or {}
    return ch.get(key, 0) if isinstance(ch, dict) else 0


def simulate_trigger(ovd, cfg):
    """
    Simulate a VSCM v4 trigger using fields already stored in the OVD JSONL.

    It cannot recover ovd_not_ok/vscm_none samples because those lack tracked
    VSCM statistics. It only changes the final decision for existing VSCM
    candidates.
    """
    vscm = ovd.get("vscm")
    if not isinstance(vscm, dict):
        return False, "vscm_none"

    motorcycle_peak = vscm.get("motorcycle_peak")
    if motorcycle_peak is None:
        motorcycle_peak = get_peak(ovd, "motorcycle")

    # Current true positives remain true unless motorcycle threshold still suppresses them.
    if vscm.get("triggered"):
        if cfg["suppress_motorcycle"] and motorcycle_peak >= cfg["motorcycle_suppress_peak"]:
            return False, "suppressed_motorcycle_v4"
        return True, "original_triggered"

    # Recover samples that were only suppressed by a single motorcycle detection.
    if (
        vscm.get("suppression_reason") == "motorcycle_evidence"
        and vscm.get("triggered_before_motorcycle_suppression")
        and motorcycle_peak < cfg["motorcycle_suppress_peak"]
    ):
        return True, "recover_motorcycle_suppressed"

    static_count = vscm.get("static_count") or 0
    moving_count = vscm.get("moving_count") or 0
    static_ratio = vscm.get("static_ratio")
    static_ratio = 999 if static_ratio is None else float(static_ratio)
    context_frames = vscm.get("context_frames") or 0
    center = vscm.get("cluster_center_x_ratio")
    center_ok = center is not None and cfg["center_min"] <= float(center) <= cfg["center_max"]
    queue_like = bool(vscm.get("queue_like"))

    person = ctx(vscm, "person")
    cone = ctx(vscm, "traffic cone")
    cv = ctx(vscm, "construction vehicle")
    barrier = ctx(vscm, "road barrier")

    # New medium-confidence context branch:
    # no explicit vehicle relation, but local static vehicles plus persistent
    # response/scene evidence. This is the only branch intended to recover
    # context_without_vehicle_relation cases.
    scene_aux = (
        cv >= cfg["context_cv"]
        or cone >= cfg["context_cone"]
        or barrier >= cfg["context_barrier"]
    )
    context_medium_conf = (
        not vscm.get("proximity")
        and not queue_like
        and cfg["static_min"] <= static_count <= cfg["static_max"]
        and static_ratio <= cfg["static_ratio_max"]
        and moving_count >= cfg["moving_min"]
        and context_frames >= cfg["context_frames_min"]
        and person >= cfg["context_person"]
        and scene_aux
        and center_ok
    )
    if context_medium_conf:
        if cfg["suppress_motorcycle"] and motorcycle_peak >= cfg["motorcycle_suppress_peak"]:
            return False, "suppressed_motorcycle_v4"
        return True, "context_medium_conf"

    return False, vscm.get("proximity_reason") or "not_triggered"


def main():
    ap = argparse.ArgumentParser(description="Offline simulation for VSCM v4 thresholds; no GPU/video decode.")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--ovd", required=True)
    ap.add_argument("--targets", default=DEFAULT_TARGETS)
    ap.add_argument("--positive-label", default="\u591a\u8f66\u4e8b\u6545")
    ap.add_argument("--static-min", type=int, default=2)
    ap.add_argument("--static-max", type=int, default=5)
    ap.add_argument("--static-ratio-max", type=float, default=0.22)
    ap.add_argument("--moving-min", type=int, default=7)
    ap.add_argument("--context-frames-min", type=int, default=6)
    ap.add_argument("--context-person", type=int, default=6)
    ap.add_argument("--context-cv", type=int, default=2)
    ap.add_argument("--context-cone", type=int, default=3)
    ap.add_argument("--context-barrier", type=int, default=8)
    ap.add_argument("--center-min", type=float, default=0.20)
    ap.add_argument("--center-max", type=float, default=0.85)
    ap.add_argument("--motorcycle-suppress-peak", type=int, default=2)
    ap.add_argument("--no-suppress-motorcycle", action="store_true")
    args = ap.parse_args()

    cfg = vars(args).copy()
    cfg["suppress_motorcycle"] = not args.no_suppress_motorcycle

    targets = parse_csv(args.targets)
    labels, totals = load_gt(args.gt)

    trigger = Counter()
    candidates = Counter()
    ok_count = Counter()
    reason = defaultdict(Counter)
    examples = defaultdict(list)

    with Path(args.ovd).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            video = obj.get("video", "")
            label = labels.get(video, video.split("_")[0] if video else "")
            if targets and label not in targets:
                continue

            if not obj.get("ok"):
                reason[label]["ovd_not_ok"] += 1
                continue

            ok_count[label] += 1
            ovd = obj.get("ovd") or {}
            if isinstance(ovd.get("vscm"), dict):
                candidates[label] += 1
            trig, why = simulate_trigger(ovd, cfg)
            reason[label][why] += 1
            if trig:
                trigger[label] += 1
                if len(examples[label]) < 8:
                    vscm = ovd.get("vscm") or {}
                    examples[label].append(
                        {
                            "video": video,
                            "reason": why,
                            "static": vscm.get("static_count"),
                            "moving": vscm.get("moving_count"),
                            "ratio": vscm.get("static_ratio"),
                            "ctx_frames": vscm.get("context_frames"),
                            "ctx": vscm.get("context_hits"),
                            "motorcycle_peak": vscm.get("motorcycle_peak", get_peak(ovd, "motorcycle")),
                        }
                    )

    print("cfg =", json.dumps({k: cfg[k] for k in [
        "static_min", "static_max", "static_ratio_max", "moving_min",
        "context_frames_min", "context_person", "context_cv",
        "context_cone", "context_barrier", "center_min", "center_max",
        "motorcycle_suppress_peak", "suppress_motorcycle"
    ]}, ensure_ascii=False))
    print()
    print("类别               总数   OVD_OK  候选   触发   全量触发率   OK内触发率   候选内触发率")
    print("-" * 86)
    for label in targets:
        total = totals.get(label, 0)
        ok = ok_count[label]
        cand = candidates[label]
        trig = trigger[label]
        print(
            f"{label:<12} {total:6d} {ok:7d} {cand:5d} {trig:5d} "
            f"{(trig / total * 100 if total else 0):9.1f}% "
            f"{(trig / ok * 100 if ok else 0):10.1f}% "
            f"{(trig / cand * 100 if cand else 0):12.1f}%"
        )

    print("\n=== trigger/reject reasons ===")
    for label in targets:
        print("\n[" + label + "]", dict(reason[label].most_common(20)))

    print("\n=== triggered examples ===")
    for label in targets:
        print("\n[" + label + "]")
        for item in examples[label]:
            print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
