import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


TARGET = "多车事故"


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


def norm_text(text):
    return "" if text is None else str(text).strip()


def is_multicar_label(label):
    text = norm_text(label)
    return text == TARGET or ("多车" in text and "事故" in text)


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
    "事故现场",
    "事故处置",
]

NEGATION_WORDS = [
    "未见",
    "没有",
    "无",
    "未发现",
    "未出现",
    "不属于",
    "非交通事故",
]

SINGLE_STOP_WORDS = [
    "一辆",
    "单车",
    "完全静止",
    "长时间静止",
    "停靠",
    "故障停车",
    "异常停车",
]


def _has_negation_before(text, pos, window=10):
    start = max(0, pos - window)
    prefix = text[start:pos]
    return any(w in prefix for w in NEGATION_WORDS)


def has_positive_collision_evidence(eaca_row):
    """Detect explicit collision evidence in clip raw text while ignoring negated mentions."""
    for clip in eaca_row.get("clips", []):
        raw = norm_text(clip.get("raw"))
        for word in POSITIVE_COLLISION_WORDS:
            start = 0
            while True:
                pos = raw.find(word, start)
                if pos < 0:
                    break
                if not _has_negation_before(raw, pos):
                    return True
                start = pos + len(word)
    return False


def positive_collision_clip_count(eaca_row):
    count = 0
    for clip in eaca_row.get("clips", []):
        raw = norm_text(clip.get("raw"))
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


def has_single_stop_without_collision(eaca_row):
    raw_all = " ".join(norm_text(clip.get("raw")) for clip in eaca_row.get("clips", []))
    return any(w in raw_all for w in SINGLE_STOP_WORDS) and not has_positive_collision_evidence(eaca_row)


def main():
    ap = argparse.ArgumentParser(description="Fuse EACA multicar signal into E8 predictions.")
    ap.add_argument("--base-pred", required=True, help="E8 prediction jsonl.")
    ap.add_argument("--eaca", required=True, help="EACA output jsonl.")
    ap.add_argument("--output", required=True, help="Output jsonl for eval_traffic_abnormal_only.py.")
    ap.add_argument("--base-pred-field", default="pred")
    ap.add_argument("--fused-field", default="pred_eaca_fused")
    ap.add_argument(
        "--signal-reasons",
        default="",
        help=(
            "Comma-separated EACA reasons allowed to trigger multicar override. "
            "Empty means accept any EACA multicar label."
        ),
    )
    ap.add_argument("--min-accident-count", type=int, default=0, help="Require at least this many ACCIDENT clips.")
    ap.add_argument(
        "--max-parking-count",
        type=int,
        default=-1,
        help="If >=0, suppress multicar signal when PARKING clip count is greater than this.",
    )
    ap.add_argument(
        "--max-congestion-count",
        type=int,
        default=-1,
        help="If >=0, suppress multicar signal when CONGESTION clip count is greater than this.",
    )
    ap.add_argument(
        "--require-accident-gt-congestion",
        action="store_true",
        help="Require ACCIDENT clip count to be strictly greater than CONGESTION clip count.",
    )
    ap.add_argument(
        "--require-accident-gt-parking",
        action="store_true",
        help="Require ACCIDENT clip count to be strictly greater than PARKING clip count.",
    )
    ap.add_argument(
        "--min-multicar-label-count",
        type=int,
        default=0,
        help="Require EACA clip_label_counts['多车事故'] to be at least this value. Useful for v2_soft fallback.",
    )
    ap.add_argument(
        "--require-collision-keyword",
        action="store_true",
        help="Require non-negated collision/追尾/受损 evidence in EACA clip raw text before multicar override.",
    )
    ap.add_argument(
        "--min-collision-clip-count",
        type=int,
        default=0,
        help="Require non-negated collision evidence to appear in at least this many clips.",
    )
    ap.add_argument(
        "--suppress-single-stop-without-collision",
        action="store_true",
        help="Suppress multicar override when clip raw text looks like single stopped vehicle and lacks collision evidence.",
    )
    args = ap.parse_args()

    base_map = by_video(read_jsonl(args.base_pred))
    eaca_rows = read_jsonl(args.eaca)
    allowed_reasons = {x.strip() for x in args.signal_reasons.split(",") if x.strip()}

    out_rows = []
    stats = Counter()
    signal_by_gt = Counter()
    changes_by_gt = Counter()
    eaca_label_counts = Counter()
    eaca_reason_counts = Counter()
    candidate_patterns = Counter()
    signal_patterns = Counter()
    missing_base = []

    for eaca in eaca_rows:
        video = eaca.get("video", "")
        base = dict(base_map.get(video, {}))
        if not base:
            missing_base.append(video)
            base = {
                "video": video,
                "gt": eaca.get("gt_label") or label_from_video(video),
                args.base_pred_field: "",
            }

        gt = label_from_video(video)
        eaca_label = eaca.get("eaca_label")
        eaca_reason = eaca.get("eaca_reason")
        counts = eaca.get("signal_counts") or {}
        label_counts = eaca.get("clip_label_counts") or {}
        accident_count = int(counts.get("accident", 0) or 0)
        collision_count = int(counts.get("collision", 0) or 0)
        close_multi_count = int(counts.get("close_multi_stop", 0) or 0)
        congestion_count = int(counts.get("congestion", 0) or 0)
        parking_count = int(counts.get("parking", 0) or 0)
        multicar_label_count = int(label_counts.get(TARGET, 0) or 0)
        eaca_label_counts[norm_text(eaca_label) or "<empty>"] += 1
        eaca_reason_counts[norm_text(eaca_reason) or "<empty>"] += 1
        multicar_candidate = is_multicar_label(eaca_label) and (not allowed_reasons or eaca_reason in allowed_reasons)
        if multicar_candidate:
            candidate_patterns[(gt, accident_count, collision_count, close_multi_count, congestion_count, parking_count, norm_text(eaca_reason))] += 1

        threshold_pass = True
        if args.min_accident_count > 0 and accident_count < args.min_accident_count:
            threshold_pass = False
        if args.max_parking_count >= 0 and parking_count > args.max_parking_count:
            threshold_pass = False
        if args.max_congestion_count >= 0 and congestion_count > args.max_congestion_count:
            threshold_pass = False
        if args.require_accident_gt_congestion and accident_count <= congestion_count:
            threshold_pass = False
        if args.require_accident_gt_parking and accident_count <= parking_count:
            threshold_pass = False
        if args.min_multicar_label_count > 0 and multicar_label_count < args.min_multicar_label_count:
            threshold_pass = False
        if args.require_collision_keyword and not has_positive_collision_evidence(eaca):
            threshold_pass = False
        collision_clip_count = positive_collision_clip_count(eaca)
        if args.min_collision_clip_count > 0 and collision_clip_count < args.min_collision_clip_count:
            threshold_pass = False
        if args.suppress_single_stop_without_collision and has_single_stop_without_collision(eaca):
            threshold_pass = False

        signal = multicar_candidate and threshold_pass
        if signal:
            signal_by_gt[gt] += 1
            signal_patterns[(gt, accident_count, collision_count, close_multi_count, congestion_count, parking_count, norm_text(eaca_reason))] += 1

        base_pred = base.get(args.base_pred_field) or ""
        fused_pred = TARGET if signal else base_pred

        rec = dict(base)
        rec[args.fused_field] = fused_pred
        rec["eaca_label"] = eaca_label
        rec["eaca_reason"] = eaca_reason
        rec["eaca_signal_multicar"] = bool(signal)
        rec["eaca_signal_counts"] = eaca.get("signal_counts", {})
        rec["eaca_clip_label_counts"] = eaca.get("clip_label_counts", {})
        rec["eaca_source"] = args.eaca
        rec["eaca_multicar_label_count"] = multicar_label_count
        rec["eaca_collision_clip_count"] = collision_clip_count

        stats["total"] += 1
        if signal:
            stats["multicar_signal"] += 1
        if signal and TARGET not in str(base_pred):
            stats["changed_to_multicar"] += 1
            changes_by_gt[gt] += 1
        out_rows.append(rec)

    write_jsonl(args.output, out_rows)

    print("output:", args.output)
    print("stats:", dict(stats))
    print("eaca_label_counts:", dict(eaca_label_counts))
    print("eaca_reason_counts:", dict(eaca_reason_counts))
    print("signal_by_gt:", dict(signal_by_gt))
    print("changes_by_gt:", dict(changes_by_gt))
    print("candidate_patterns(gt, accident, collision, close_multi, congestion, parking, reason):", {str(k): v for k, v in candidate_patterns.items()})
    print("signal_patterns(gt, accident, collision, close_multi, congestion, parking, reason):", {str(k): v for k, v in signal_patterns.items()})
    if missing_base:
        print("missing_base:", len(missing_base), missing_base[:10])


if __name__ == "__main__":
    main()
