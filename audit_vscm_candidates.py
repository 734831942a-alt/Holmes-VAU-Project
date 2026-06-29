import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


VEHICLE_LABELS = ("car", "truck", "bus")
CONTEXT_LABELS = ("person", "traffic cone", "construction vehicle", "road barrier")


def parse_csv(text):
    return [x.strip() for x in (text or "").split(",") if x.strip()]


def load_gt(path):
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


def safe_counts(ovd, key):
    value = ovd.get(key)
    return value if isinstance(value, dict) else {}


def feature_summary(ovd):
    sum_count = safe_counts(ovd, "sum_count")
    peak_count = safe_counts(ovd, "peak_count")
    first_frame = safe_counts(ovd, "first_frame")
    last_frame = safe_counts(ovd, "last_frame")

    vehicle_sum = sum(sum_count.get(k, 0) for k in VEHICLE_LABELS)
    vehicle_peak = max([peak_count.get(k, 0) for k in VEHICLE_LABELS] or [0])
    context_sum = sum(sum_count.get(k, 0) for k in CONTEXT_LABELS)
    context_peak = max([peak_count.get(k, 0) for k in CONTEXT_LABELS] or [0])
    motorcycle_peak = peak_count.get("motorcycle", 0)

    return {
        "sampled_frames": ovd.get("sampled_frames", 0),
        "vehicle_sum": vehicle_sum,
        "vehicle_peak": vehicle_peak,
        "context_sum": context_sum,
        "context_peak": context_peak,
        "motorcycle_peak": motorcycle_peak,
        "person_sum": sum_count.get("person", 0),
        "person_peak": peak_count.get("person", 0),
        "construction_vehicle_sum": sum_count.get("construction vehicle", 0),
        "construction_vehicle_peak": peak_count.get("construction vehicle", 0),
        "traffic_cone_sum": sum_count.get("traffic cone", 0),
        "traffic_cone_peak": peak_count.get("traffic cone", 0),
        "road_barrier_sum": sum_count.get("road barrier", 0),
        "road_barrier_peak": peak_count.get("road barrier", 0),
        "vehicle_first": min([first_frame[k] for k in VEHICLE_LABELS if k in first_frame], default=None),
        "vehicle_last": max([last_frame[k] for k in VEHICLE_LABELS if k in last_frame], default=None),
    }


def bucket_vscm_none(ovd):
    # This is approximate. Existing OVD summaries do not store per-frame boxes
    # or early-return reasons inside _compute_vscm_v3.
    f = feature_summary(ovd)

    if f["sampled_frames"] <= 0:
        return "A_no_sampled_frames"
    if f["vehicle_sum"] == 0:
        return "B_no_vehicle_detection"
    if f["vehicle_peak"] < 2:
        return "C_too_few_vehicles_per_frame"
    if f["vehicle_sum"] < 5:
        return "D_low_vehicle_total"
    if f["context_sum"] == 0:
        return "E_no_context_evidence"
    if f["person_sum"] == 0 and f["construction_vehicle_sum"] == 0:
        return "F_no_person_or_construction"
    return "G_likely_track_or_static_filter"


def bucket_not_triggered(vscm, ovd):
    ch = vscm.get("context_hits") or {}
    reasons = []

    if not vscm.get("proximity"):
        reasons.append("no_relation_evidence")
    if (vscm.get("context_frames") or 0) < 6:
        reasons.append("context_frames_lt_6")
    if ch.get("person", 0) < 6:
        reasons.append("person_lt_6")
    if not (
        ch.get("construction vehicle", 0) >= 3
        or ch.get("traffic cone", 0) >= 3
        or ch.get("road barrier", 0) >= 6
    ):
        reasons.append("scene_aux_weak")
    if not (2 <= (vscm.get("static_count") or 0) <= 4):
        reasons.append("static_count_out_2_4")
    if (vscm.get("static_ratio") if vscm.get("static_ratio") is not None else 999) > 0.20:
        reasons.append("static_ratio_gt_0.20")
    if (vscm.get("moving_count") or 0) < 7:
        reasons.append("moving_count_lt_7")
    center = vscm.get("cluster_center_x_ratio")
    if center is None or not (0.20 <= center <= 0.85):
        reasons.append("center_out_0.20_0.85")
    if vscm.get("queue_like"):
        reasons.append("queue_like")

    peak_count = safe_counts(ovd, "peak_count")
    if peak_count.get("motorcycle", 0) >= 1 or vscm.get("suppression_reason") == "motorcycle_evidence":
        reasons.append("motorcycle_suppressed")

    return "+".join(reasons) if reasons else "unknown_not_triggered"


def compact_info(ovd, vscm):
    f = feature_summary(ovd)
    info = dict(f)
    if isinstance(vscm, dict):
        for k in [
            "static_count",
            "moving_count",
            "static_ratio",
            "cluster_center_x_ratio",
            "proximity",
            "proximity_reason",
            "contact_frames",
            "close_frames",
            "context_frames",
            "context_hits",
            "meaningful_scene_context",
            "scene_only_high_conf",
            "queue_scene_high_conf",
            "queue_like_frames",
            "checked_frames",
            "queue_like",
            "suppression_reason",
            "triggered",
        ]:
            info[k] = vscm.get(k)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--ovd", required=True)
    ap.add_argument("--targets", default="\u591a\u8f66\u4e8b\u6545,\u62e5\u5835,\u5f02\u5e38\u505c\u8f66,\u5360\u9053\u65bd\u5de5,\u4e8c\u8f6e\u8f66\u8f86\u95ef\u5165,\u629b\u6d12\u7269")
    ap.add_argument("--examples-per-bucket", type=int, default=5)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    targets = parse_csv(args.targets)
    labels, totals = load_gt(args.gt)

    stage = defaultdict(Counter)
    none_buckets = defaultdict(Counter)
    not_trigger_buckets = defaultdict(Counter)
    examples = defaultdict(lambda: defaultdict(list))
    csv_rows = []

    rows = 0
    for line in Path(args.ovd).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows += 1
        obj = json.loads(line)
        video = obj.get("video", "")
        label = labels.get(video, video.split("_")[0] if video else "")
        if targets and label not in targets:
            continue

        if not obj.get("ok"):
            stage[label]["not_ok"] += 1
            bucket = "not_ok:" + str(obj.get("err", "unknown"))
            none_buckets[label][bucket] += 1
            if len(examples[label][bucket]) < args.examples_per_bucket:
                examples[label][bucket].append((video, {"err": obj.get("err")}))
            continue

        ovd = obj.get("ovd") or {}
        vscm = ovd.get("vscm")
        stage[label]["ok"] += 1

        if vscm is None:
            stage[label]["vscm_none"] += 1
            bucket = bucket_vscm_none(ovd)
            none_buckets[label][bucket] += 1
            info = compact_info(ovd, None)
            if len(examples[label][bucket]) < args.examples_per_bucket:
                examples[label][bucket].append((video, info))
            csv_rows.append({"label": label, "video": video, "stage": "vscm_none", "bucket": bucket, **info})
            continue

        stage[label]["vscm_candidate"] += 1
        if vscm.get("triggered"):
            stage[label]["triggered"] += 1
            bucket = "triggered:" + str(vscm.get("proximity_reason") or "none_reason")
        else:
            stage[label]["not_triggered"] += 1
            bucket = bucket_not_triggered(vscm, ovd)
            not_trigger_buckets[label][bucket] += 1

        info = compact_info(ovd, vscm)
        if len(examples[label][bucket]) < args.examples_per_bucket:
            examples[label][bucket].append((video, info))
        csv_rows.append({"label": label, "video": video, "stage": "candidate", "bucket": bucket, **info})

    print("records_read =", rows)
    print("\n=== stage summary ===")
    print("label total ok vscm_none candidate triggered not_triggered")
    for label in targets:
        c = stage[label]
        print(
            label,
            totals.get(label, 0),
            c.get("ok", 0),
            c.get("vscm_none", 0),
            c.get("vscm_candidate", 0),
            c.get("triggered", 0),
            c.get("not_triggered", 0),
        )

    print("\n=== vscm_none approximate buckets ===")
    for label in targets:
        print("\n[" + label + "]")
        for bucket, count in none_buckets[label].most_common():
            print(bucket, count)

    print("\n=== candidate_not_triggered buckets ===")
    for label in targets:
        print("\n[" + label + "]")
        for bucket, count in not_trigger_buckets[label].most_common(20):
            print(bucket, count)

    print("\n=== examples ===")
    for label in targets:
        print("\n##", label)
        for bucket, items in examples[label].items():
            print("\n--", bucket)
            for video, info in items:
                print(video, json.dumps(info, ensure_ascii=False, default=str))

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({k for row in csv_rows for k in row})
        with out.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print("\nsaved csv:", out)


if __name__ == "__main__":
    main()
