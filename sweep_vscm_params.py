import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_csv(text):
    return [x.strip() for x in (text or "").split(",") if x.strip()]


def parse_floats(text):
    return [float(x) for x in parse_csv(text)]


def parse_ints(text):
    return [int(x) for x in parse_csv(text)]


def load_labels(path):
    labels = {}
    totals = Counter()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        video = obj.get("video", "")
        label = video.split("_")[0] if video else ""
        labels[video] = label
        totals[label] += 1
    return labels, totals


def iter_records(ovd_path, labels):
    for line in Path(ovd_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        video = obj.get("video", "")
        label = labels.get(video, video.split("_")[0] if video else "")
        ovd = obj.get("ovd") or {}
        yield label, video, ovd, ovd.get("vscm")


def aux_ok(ch, cv_min, cone_min, barrier_min):
    return (
        ch.get("construction vehicle", 0) >= cv_min
        or ch.get("traffic cone", 0) >= cone_min
        or ch.get("road barrier", 0) >= barrier_min
    )


def meaningful_ok(ch, person_min, cv_min, cone_min):
    # Road barrier is deliberately excluded as standalone evidence; it is too
    # common in highway scenes.
    return (
        ch.get("person", 0) >= person_min
        or (
            ch.get("construction vehicle", 0) >= cv_min
            and ch.get("person", 0) >= 1
        )
        or (
            ch.get("traffic cone", 0) >= cone_min
            and (
                ch.get("person", 0) >= 1
                or ch.get("construction vehicle", 0) >= 2
            )
        )
    )


def simulate(vscm, ovd, cfg):
    if not isinstance(vscm, dict):
        return False

    ch = vscm.get("context_hits") or {}
    static_count = vscm.get("static_count") or 0
    moving_count = vscm.get("moving_count") or 0
    static_ratio = vscm.get("static_ratio")
    static_ratio = 999.0 if static_ratio is None else float(static_ratio)
    context_frames = vscm.get("context_frames") or 0
    center = vscm.get("cluster_center_x_ratio")
    center = -1.0 if center is None else float(center)
    queue_like = bool(vscm.get("queue_like"))
    relation_evidence = bool(vscm.get("proximity"))

    peak_count = ovd.get("peak_count") or {}
    if cfg["suppress_motorcycle"] and (peak_count.get("motorcycle", 0) >= cfg["motorcycle_peak"]):
        return False

    rel_scene = (
        relation_evidence
        and context_frames >= cfg["rel_context_frames"]
        and meaningful_ok(
            ch,
            cfg["meaningful_person"],
            cfg["meaningful_cv"],
            cfg["meaningful_cone"],
        )
        and not queue_like
    )

    scene_only = (
        ch.get("person", 0) >= cfg["scene_person"]
        and aux_ok(ch, cfg["scene_cv"], cfg["scene_cone"], cfg["scene_barrier"])
        and context_frames >= cfg["scene_context_frames"]
        and cfg["static_min"] <= static_count <= cfg["static_max"]
        and static_ratio <= cfg["scene_static_ratio"]
        and moving_count >= cfg["scene_moving"]
        and not queue_like
        and cfg["center_min"] <= center <= cfg["center_max"]
    )

    queue_scene = (
        queue_like
        and ch.get("person", 0) >= cfg["queue_person"]
        and ch.get("construction vehicle", 0) >= cfg["queue_cv"]
        and context_frames >= cfg["queue_context_frames"]
        and cfg["static_min"] <= static_count <= cfg["static_max"]
        and static_ratio <= cfg["queue_static_ratio"]
        and moving_count >= cfg["queue_moving"]
        and cfg["center_min"] <= center <= cfg["center_max"]
    )

    return rel_scene or scene_only or queue_scene


def evaluate(records, totals, positive_label, false_labels, cfg):
    trigger = Counter()
    candidates = Counter()
    examples = defaultdict(list)

    for label, video, ovd, vscm in records:
        if isinstance(vscm, dict):
            candidates[label] += 1
        if simulate(vscm, ovd, cfg):
            trigger[label] += 1
            if len(examples[label]) < 3:
                examples[label].append(video)

    pos_total = totals.get(positive_label, 0)
    pos_cand = candidates.get(positive_label, 0)
    pos_trig = trigger.get(positive_label, 0)
    pos_all_rate = pos_trig / max(pos_total, 1)
    pos_cand_rate = pos_trig / max(pos_cand, 1)

    false_all = 0
    false_cand_rate_max = 0.0
    false_all_rate_max = 0.0
    false_detail = {}
    for label in false_labels:
        t = trigger.get(label, 0)
        total = totals.get(label, 0)
        cand = candidates.get(label, 0)
        all_rate = t / max(total, 1)
        cand_rate = t / max(cand, 1)
        false_all += t
        false_all_rate_max = max(false_all_rate_max, all_rate)
        false_cand_rate_max = max(false_cand_rate_max, cand_rate)
        false_detail[label] = (total, cand, t, all_rate, cand_rate)

    return {
        "positive": (pos_total, pos_cand, pos_trig, pos_all_rate, pos_cand_rate),
        "false_all": false_all,
        "false_all_rate_max": false_all_rate_max,
        "false_cand_rate_max": false_cand_rate_max,
        "false_detail": false_detail,
        "examples": dict(examples),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--ovd", required=True)
    ap.add_argument("--positive-label", default="\u591a\u8f66\u4e8b\u6545")
    ap.add_argument(
        "--false-labels",
        default="\u62e5\u5835,\u5f02\u5e38\u505c\u8f66,\u5360\u9053\u65bd\u5de5,\u4e8c\u8f6e\u8f66\u8f86\u95ef\u5165,\u629b\u6d12\u7269",
    )
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-false-all-rate", type=float, default=0.01)
    ap.add_argument("--max-false-cand-rate", type=float, default=0.05)
    ap.add_argument("--scene-person", default="4,5,6")
    ap.add_argument("--scene-cv", default="2,3")
    ap.add_argument("--scene-cone", default="3,5")
    ap.add_argument("--scene-barrier", default="6,8,10")
    ap.add_argument("--scene-context-frames", default="4,5,6")
    ap.add_argument("--scene-static-ratio", default="0.20,0.22,0.24")
    ap.add_argument("--scene-moving", default="5,7")
    ap.add_argument("--queue-person", default="6,8,10")
    ap.add_argument("--queue-cv", default="2,3")
    ap.add_argument("--queue-context-frames", default="6,8")
    ap.add_argument("--queue-static-ratio", default="0.16,0.20,0.22")
    ap.add_argument("--queue-moving", default="7,10")
    ap.add_argument("--center-min", type=float, default=0.20)
    ap.add_argument("--center-max", type=float, default=0.85)
    ap.add_argument("--static-min", type=int, default=2)
    ap.add_argument("--static-max", type=int, default=4)
    ap.add_argument("--motorcycle-peak", type=int, default=1)
    ap.add_argument("--no-suppress-motorcycle", action="store_true")
    args = ap.parse_args()

    false_labels = parse_csv(args.false_labels)
    labels, totals = load_labels(args.gt)
    records = list(iter_records(args.ovd, labels))

    base = {
        "rel_context_frames": 2,
        "meaningful_person": 2,
        "meaningful_cv": 3,
        "meaningful_cone": 3,
        "center_min": args.center_min,
        "center_max": args.center_max,
        "static_min": args.static_min,
        "static_max": args.static_max,
        "motorcycle_peak": args.motorcycle_peak,
        "suppress_motorcycle": not args.no_suppress_motorcycle,
    }

    grids = {
        "scene_person": parse_ints(args.scene_person),
        "scene_cv": parse_ints(args.scene_cv),
        "scene_cone": parse_ints(args.scene_cone),
        "scene_barrier": parse_ints(args.scene_barrier),
        "scene_context_frames": parse_ints(args.scene_context_frames),
        "scene_static_ratio": parse_floats(args.scene_static_ratio),
        "scene_moving": parse_ints(args.scene_moving),
        "queue_person": parse_ints(args.queue_person),
        "queue_cv": parse_ints(args.queue_cv),
        "queue_context_frames": parse_ints(args.queue_context_frames),
        "queue_static_ratio": parse_floats(args.queue_static_ratio),
        "queue_moving": parse_ints(args.queue_moving),
    }

    keys = list(grids)
    results = []
    for values in itertools.product(*(grids[k] for k in keys)):
        cfg = dict(base)
        cfg.update(dict(zip(keys, values)))
        ev = evaluate(records, totals, args.positive_label, false_labels, cfg)
        if ev["false_all_rate_max"] <= args.max_false_all_rate and ev["false_cand_rate_max"] <= args.max_false_cand_rate:
            results.append((ev["positive"][4], ev["positive"][3], -ev["false_all"], cfg, ev))

    results.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))

    print("records =", len(records))
    print("positive =", args.positive_label, "false_labels =", false_labels)
    print("accepted_configs =", len(results))
    print("limits: max_false_all_rate <=", args.max_false_all_rate, "max_false_cand_rate <=", args.max_false_cand_rate)

    for rank, (_, _, _, cfg, ev) in enumerate(results[: args.top_k], 1):
        pt, pc, ptr, pall, pcand = ev["positive"]
        print("\n== rank", rank, "==")
        print(
            "positive total/cand/trig/all_rate/cand_rate =",
            pt,
            pc,
            ptr,
            f"{pall:.4f}",
            f"{pcand:.4f}",
        )
        print(
            "false_all =",
            ev["false_all"],
            "max_false_all_rate =",
            f"{ev['false_all_rate_max']:.4f}",
            "max_false_cand_rate =",
            f"{ev['false_cand_rate_max']:.4f}",
        )
        print("cfg =", json.dumps(cfg, ensure_ascii=False, sort_keys=True))
        for label, detail in ev["false_detail"].items():
            total, cand, trig, all_rate, cand_rate = detail
            if trig:
                print("false", label, "total/cand/trig/all/cand =", total, cand, trig, f"{all_rate:.4f}", f"{cand_rate:.4f}")
        print("positive_examples =", ev["examples"].get(args.positive_label, []))


if __name__ == "__main__":
    main()
