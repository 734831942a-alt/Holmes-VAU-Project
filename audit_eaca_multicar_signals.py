import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


TARGET = "多车事故"

POSITIVE_COLLISION_WORDS = [
    "追尾碰撞",
    "发生追尾",
    "发生碰撞",
    "相撞",
    "碰撞",
    "追尾",
    "剐蹭",
    "刮擦",
    "车头受损",
    "明显受损",
    "受损",
    "事故现场",
    "事故处置",
]
NEGATION_WORDS = ["未见", "没有", "无", "未发现", "未出现", "不属于", "非交通事故"]
SINGLE_STOP_WORDS = ["一辆", "单车", "完全静止", "长时间静止", "停靠", "故障停车", "异常停车", "应急车道"]
CONGESTION_WORDS = ["拥堵", "排队", "缓行", "低速", "车流密集", "减速", "绕行", "变道"]
UNCERTAIN_WORDS = ["疑似", "可能", "推测", "嫌疑", "未见明显", "无明显"]
CONSTRUCTION_WORDS = ["施工", "锥桶", "围挡", "导流", "工程车", "作业车"]
TWOWHEEL_WORDS = ["二轮", "摩托", "电动车", "非机动车"]
DEBRIS_WORDS = ["抛洒", "抛撒", "散落", "异物", "障碍物", "货物散落"]


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def by_video(rows):
    out = {}
    for row in rows:
        video = row.get("video")
        if video:
            out[video] = row
    return out


def label_from_video(video):
    return str(video or "").split("_")[0]


def norm_text(text):
    return "" if text is None else str(text).strip()


def has_negation_before(text, pos, window=10):
    return any(w in text[max(0, pos - window):pos] for w in NEGATION_WORDS)


def count_nonnegated(text, words):
    count = 0
    hits = []
    for word in words:
        start = 0
        while True:
            pos = text.find(word, start)
            if pos < 0:
                break
            neg = has_negation_before(text, pos)
            hits.append((word, neg))
            if not neg:
                count += 1
            start = pos + len(word)
    return count, hits


def count_any(text, words):
    return sum(text.count(w) for w in words)


def clip_features(clip):
    raw = norm_text(clip.get("raw"))
    collision_count, collision_hits = count_nonnegated(raw, POSITIVE_COLLISION_WORDS)
    return {
        "collision_count": collision_count,
        "collision_hits": collision_hits,
        "single_stop_count": count_any(raw, SINGLE_STOP_WORDS),
        "congestion_count": count_any(raw, CONGESTION_WORDS),
        "uncertain_count": count_any(raw, UNCERTAIN_WORDS),
        "construction_count": count_any(raw, CONSTRUCTION_WORDS),
        "twowheel_count": count_any(raw, TWOWHEEL_WORDS),
        "debris_count": count_any(raw, DEBRIS_WORDS),
    }


def row_features(row):
    clips = row.get("clips", [])
    clip_feats = [clip_features(c) for c in clips]
    collision_clip_count = sum(1 for f in clip_feats if f["collision_count"] > 0)
    total = Counter()
    hits = []
    for f in clip_feats:
        for key, val in f.items():
            if key == "collision_hits":
                hits.extend(val)
            else:
                total[key] += val
    return {
        "collision_clip_count": collision_clip_count,
        "counts": dict(total),
        "collision_hits": hits,
        "signal_counts": row.get("signal_counts") or {},
        "clip_label_counts": row.get("clip_label_counts") or {},
    }


def detect_base_label(text):
    labels = ["多车事故", "拥堵", "异常停车", "占道施工", "二轮车辆闯入", "抛洒物"]
    text = norm_text(text)
    best_pos = -1
    best = None
    for label in labels:
        pos = text.rfind(label)
        if pos > best_pos:
            best_pos = pos
            best = label
    return best


def compact(text, n=180):
    text = " ".join(norm_text(text).split())
    return text if len(text) <= n else text[:n] + "..."


def main():
    ap = argparse.ArgumentParser(description="Audit EACA multicar TP/FP signal features.")
    ap.add_argument("--eaca", required=True)
    ap.add_argument("--base-pred", default="")
    ap.add_argument("--base-pred-field", default="pred")
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--show-examples", type=int, default=20)
    args = ap.parse_args()

    rows = read_jsonl(args.eaca)
    base_map = by_video(read_jsonl(args.base_pred)) if args.base_pred else {}

    groups = defaultdict(list)
    for row in rows:
        if row.get("eaca_label") != args.target:
            continue
        video = row.get("video", "")
        gt = row.get("gt_label") or label_from_video(video)
        base = base_map.get(video, {})
        feat = row_features(row)
        item = {
            "video": video,
            "gt": gt,
            "is_tp": gt == args.target,
            "eaca_reason": row.get("eaca_reason"),
            "base_label": detect_base_label(base.get(args.base_pred_field, "")),
            "base_pred": base.get(args.base_pred_field, ""),
            "features": feat,
            "row": row,
        }
        groups["TP" if item["is_tp"] else "FP"].append(item)

    print("=== counts ===")
    print({k: len(v) for k, v in groups.items()})

    print("\n=== feature summary ===")
    for group_name in ["TP", "FP"]:
        items = groups.get(group_name, [])
        print(f"\n[{group_name}] n={len(items)}")
        dist = Counter()
        base_dist = Counter()
        reason_dist = Counter()
        gt_dist = Counter()
        hit_dist = Counter()
        for it in items:
            feat = it["features"]
            c = feat["counts"]
            dist[(
                feat["collision_clip_count"],
                c.get("single_stop_count", 0),
                c.get("congestion_count", 0),
                c.get("uncertain_count", 0),
                c.get("construction_count", 0),
                c.get("twowheel_count", 0),
                c.get("debris_count", 0),
            )] += 1
            base_dist[it["base_label"]] += 1
            reason_dist[it["eaca_reason"]] += 1
            gt_dist[it["gt"]] += 1
            for word, neg in feat["collision_hits"]:
                hit_dist[(word, neg)] += 1
        print("gt_dist:", dict(gt_dist))
        print("base_label_dist:", dict(base_dist))
        print("reason_dist:", dict(reason_dist))
        print("feature_tuple_dist(collision_clips,single,congestion,uncertain,construction,twowheel,debris):")
        for k, v in dist.most_common():
            print(" ", k, "=>", v)
        print("collision_hits(word, negated):", {str(k): v for k, v in hit_dist.most_common()})

    print("\n=== examples ===")
    shown = 0
    for group_name in ["TP", "FP"]:
        for it in groups.get(group_name, []):
            if shown >= args.show_examples:
                return
            shown += 1
            print("=" * 100)
            print("group:", group_name, "gt:", it["gt"], "base_label:", it["base_label"], "reason:", it["eaca_reason"])
            print("video:", it["video"])
            print("features:", it["features"])
            print("base_pred:", compact(it["base_pred"]))
            for clip in it["row"].get("clips", []):
                print("- clip", clip.get("clip_id"), "parsed:", clip.get("parsed"))
                print(compact(clip.get("raw"), 400))


if __name__ == "__main__":
    main()
