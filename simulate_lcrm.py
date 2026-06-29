import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_TARGETS = "多车事故,拥堵,异常停车,占道施工,二轮车辆闯入,抛洒物"


def parse_csv(text):
    return [x.strip() for x in (text or "").split(",") if x.strip()]


def load_gt(path):
    labels = {}
    totals = Counter()
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            video = obj.get("video", "")
            label = video.split("_", 1)[0] if "_" in video else ""
            labels[video] = label
            totals[label] += 1
    return labels, totals


def get_lcrm_record(obj):
    ovd = obj.get("ovd") or {}
    lcrm = ovd.get("lcrm")
    return lcrm if isinstance(lcrm, dict) else None


def flag_scores(lcrm, args):
    if not lcrm or not lcrm.get("ok"):
        return {}
    return {
        "local_pair": float(lcrm.get("local_pair_score") or 0.0) >= args.local_pair_thr,
        "isolated_stop": float(lcrm.get("isolated_stop_score") or 0.0) >= args.isolated_stop_thr,
        "distributed_congestion": (
            float(lcrm.get("distributed_congestion_score") or 0.0) >= args.distributed_congestion_thr
        ),
    }


def compact_lcrm(lcrm):
    keys = [
        "primary_signal",
        "local_pair_score",
        "isolated_stop_score",
        "distributed_congestion_score",
        "track_count",
        "static_count",
        "slow_count",
        "moving_count",
        "global_slow_ratio",
        "vehicle_peak",
        "slow_region_count",
        "dominant_slow_region_ratio",
        "min_static_pair_edge_gap_norm",
        "min_static_pair_edge_gap_width",
        "max_pair_close_frames",
        "max_pair_contact_frames",
        "best_static_pair",
    ]
    return {k: lcrm.get(k) for k in keys if k in lcrm}


def main():
    ap = argparse.ArgumentParser(description="Offline analysis for LCRM signals stored in OVD summary.")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--ovd", required=True)
    ap.add_argument("--targets", default=DEFAULT_TARGETS)
    ap.add_argument("--local-pair-thr", type=float, default=0.55)
    ap.add_argument("--isolated-stop-thr", type=float, default=0.55)
    ap.add_argument("--distributed-congestion-thr", type=float, default=0.55)
    ap.add_argument("--examples-per-class", type=int, default=8)
    args = ap.parse_args()

    targets = parse_csv(args.targets)
    labels, totals = load_gt(args.gt)

    ok = Counter()
    has_lcrm = Counter()
    flags = defaultdict(Counter)
    primary = defaultdict(Counter)
    score_sums = defaultdict(lambda: Counter())
    examples = defaultdict(lambda: defaultdict(list))
    missing_reason = defaultdict(Counter)

    with Path(args.ovd).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            video = obj.get("video", "")
            label = labels.get(video, video.split("_", 1)[0] if "_" in video else "")
            if targets and label not in targets:
                continue

            if obj.get("ok"):
                ok[label] += 1
            lcrm = get_lcrm_record(obj)
            if not lcrm:
                missing_reason[label]["missing_lcrm"] += 1
                continue
            if not lcrm.get("ok"):
                missing_reason[label][lcrm.get("reason", "lcrm_not_ok")] += 1
                continue

            has_lcrm[label] += 1
            primary[label][lcrm.get("primary_signal", "none")] += 1
            for key in ["local_pair_score", "isolated_stop_score", "distributed_congestion_score"]:
                score_sums[label][key] += float(lcrm.get(key) or 0.0)
            fs = flag_scores(lcrm, args)
            for signal, value in fs.items():
                if value:
                    flags[label][signal] += 1
                    if len(examples[label][signal]) < args.examples_per_class:
                        examples[label][signal].append({
                            "video": video,
                            **compact_lcrm(lcrm),
                        })

    print("thresholds =", {
        "local_pair": args.local_pair_thr,
        "isolated_stop": args.isolated_stop_thr,
        "distributed_congestion": args.distributed_congestion_thr,
    })
    print()
    print("类别               total   OVD_OK  LCRM_OK  local_pair  isolated  congestion")
    print("-" * 86)
    for label in targets:
        total = totals.get(label, 0)
        l_ok = has_lcrm[label]
        print(
            f"{label:<12} {total:7d} {ok[label]:7d} {l_ok:8d} "
            f"{flags[label]['local_pair']:11d} {flags[label]['isolated_stop']:9d} "
            f"{flags[label]['distributed_congestion']:11d}"
        )

    print("\n=== rates within LCRM_OK ===")
    for label in targets:
        denom = has_lcrm[label] or 1
        print(label, {
            "local_pair": round(flags[label]["local_pair"] / denom, 3),
            "isolated_stop": round(flags[label]["isolated_stop"] / denom, 3),
            "distributed_congestion": round(flags[label]["distributed_congestion"] / denom, 3),
        })

    print("\n=== mean scores ===")
    for label in targets:
        denom = has_lcrm[label] or 1
        print(label, {
            "local_pair": round(score_sums[label]["local_pair_score"] / denom, 3),
            "isolated_stop": round(score_sums[label]["isolated_stop_score"] / denom, 3),
            "distributed_congestion": round(score_sums[label]["distributed_congestion_score"] / denom, 3),
        })

    print("\n=== primary signals ===")
    for label in targets:
        print(label, dict(primary[label].most_common()))

    print("\n=== missing/not-ok reasons ===")
    for label in targets:
        if missing_reason[label]:
            print(label, dict(missing_reason[label].most_common()))

    print("\n=== examples ===")
    for label in targets:
        print("\n##", label)
        for signal in ["local_pair", "isolated_stop", "distributed_congestion"]:
            print("\n", signal)
            for item in examples[label][signal]:
                print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
