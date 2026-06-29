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


def main():
    ap = argparse.ArgumentParser(description="Audit Regional Motion Contrast Module signals.")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--ovd", required=True)
    ap.add_argument("--targets", default=DEFAULT_TARGETS)
    ap.add_argument("--examples", type=int, default=8)
    args = ap.parse_args()

    targets = parse_csv(args.targets)
    labels, totals = load_labels(args.gt)
    signal_counts = defaultdict(Counter)
    suppression_counts = defaultdict(Counter)
    ok_counts = Counter()
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
            rmcm = ((obj.get("ovd") or {}).get("rmcm") or {})
            if not rmcm.get("ok"):
                signal_counts[label]["rmcm_not_ok"] += 1
                continue
            ok_counts[label] += 1
            signal = rmcm.get("primary_signal") or "none"
            signal_counts[label][signal] += 1
            if rmcm.get("suppression_reason"):
                suppression_counts[label][rmcm.get("suppression_reason")] += 1
            if signal != "none" and len(examples[(label, signal)]) < args.examples:
                examples[(label, signal)].append({
                    "video": video,
                    "mode": rmcm.get("mode"),
                    "best_window_id": rmcm.get("best_window_id"),
                    "raw_primary_signal": rmcm.get("raw_primary_signal"),
                    "suppression_reason": rmcm.get("suppression_reason"),
                    "semantic_peak": rmcm.get("semantic_peak"),
                    "center_static": rmcm.get("center_static_count"),
                    "center_slow": rmcm.get("center_slow_count"),
                    "center_moving": rmcm.get("center_moving_count"),
                    "motion_contrast": rmcm.get("motion_contrast"),
                    "global_slow_ratio": rmcm.get("global_slow_ratio"),
                    "center_slow_ratio": rmcm.get("center_slow_ratio"),
                    "pair_stagnation_frames": rmcm.get("pair_stagnation_frames"),
                    "static_pair_related": rmcm.get("static_pair_related"),
                    "primary_signal": signal,
                    "windows": rmcm.get("windows"),
                })

    print("类别               总数   RMCM_OK  single_stop  multi_stag  congestion  none")
    print("-" * 78)
    for label in targets:
        c = signal_counts[label]
        total = totals.get(label, 0)
        ok = ok_counts[label]
        print(
            f"{label:<12} {total:6d} {ok:8d} "
            f"{c.get('single_vehicle_stop', 0):11d} "
            f"{c.get('multi_vehicle_stagnation', 0):11d} "
            f"{c.get('global_congestion', 0):11d} "
            f"{c.get('none', 0):5d}"
        )

    print("\n=== signal rates within RMCM_OK ===")
    for label in targets:
        ok = max(ok_counts[label], 1)
        c = signal_counts[label]
        print(
            label,
            {
                "single": round(c.get("single_vehicle_stop", 0) / ok, 3),
                "multi": round(c.get("multi_vehicle_stagnation", 0) / ok, 3),
                "congestion": round(c.get("global_congestion", 0) / ok, 3),
            },
        )

    print("\n=== suppression counts ===")
    for label in targets:
        if suppression_counts[label]:
            print(label, dict(suppression_counts[label]))

    print("\n=== examples ===")
    for key in sorted(examples):
        print("\n", key)
        for item in examples[key]:
            print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
