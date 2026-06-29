import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from eval_traffic_fixed import UNKNOWN_LABEL, detect_label, get_gt_label
from postprocess_lcrm_lite_fusion import get_ovd_payload, lcrm_lite_signal, read_jsonl


def parse_manual_txt(path):
    rows = []
    text = Path(path).read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        match = re.match(r"^(\d+)\.(\d+)\s*(.*)$", line)
        if not match:
            continue
        rows.append(
            {
                "idx": int(match.group(1)),
                "event_id": match.group(2),
                "manual_note": match.group(3).strip(),
            }
        )
    return rows


def by_event_id(rows):
    out = {}
    for row in rows:
        video = str(row.get("video") or row.get("video_path") or "")
        stem = Path(video).stem
        ids = re.findall(r"\d{6,}", stem)
        for event_id in ids:
            out.setdefault(event_id, row)
    return out


def pick_text(row, field):
    if not isinstance(row, dict):
        return ""
    for key in [field, "pred", "prediction", "output", "response", "answer"]:
        if key in row:
            return str(row.get(key) or "")
    return ""


def brief(text, limit=180):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text[:limit]


def has(text, *keywords):
    return any(keyword in text for keyword in keywords)


def annotation_bucket(note):
    if has(note, "无法打开"):
        return "unable_open"
    if has(note, "有碰撞过程"):
        return "collision_process_weak" if has(note, "不明显") else "collision_process_clear"
    if has(note, "无碰撞过程", "无明显碰撞过程"):
        if has(note, "明显碰撞痕迹", "损毁严重"):
            return "no_process_but_damage_trace"
        if has(note, "无人员在车旁"):
            return "no_process_parking_no_person"
        return "no_process_parking_with_person"
    if has(note, "看不到事故可疑车辆"):
        return "cannot_see_suspect_vehicle"
    return "other"


def note_flags(note):
    flags = []
    checks = [
        ("person_present", ("有人员在车旁", "有人员下车")),
        ("explicit_no_person", ("无人员在车旁",)),
        ("low_quality", ("画面不清晰", "极不清晰", "质量很差")),
        ("edge_or_small", ("画面边缘", "所占画面小", "所占画面较小")),
        ("occluded", ("遮挡", "挡住")),
        ("drive_away", ("开走",)),
        ("far_apart", ("相隔远",)),
        ("damage_trace", ("明显碰撞痕迹", "损毁严重")),
    ]
    for name, keywords in checks:
        if has(note, *keywords):
            flags.append(name)
    return flags


def detailed_note_flags(note):
    checks = [
        ("two_or_more_close", ("两车贴近", "三车贴近", "相隔较近")),
        ("far_or_unclear_relation", ("相隔较远", "相隔远", "相隔有一定距离")),
        ("person_handling", ("查看车辆碰撞情况", "交涉", "拍照")),
        ("explicit_evidence", ("存在事故证据", "有事故证据", "明显车损", "碰撞痕迹", "损毁严重")),
        ("no_evidence", ("无明显事故证据", "没什么特征", "看不到事故可疑车辆")),
        ("single_vehicle", ("只看到单车", "单车")),
        ("low_quality", ("画面不清晰", "极不清晰", "模糊", "质量很差")),
        ("small_or_edge", ("所占画面小", "所占画面较小", "画面边缘", "只露出一部分")),
        ("occluded", ("遮挡", "挡住")),
        ("drive_away", ("开走",)),
        ("overlap_angle", ("重合", "重叠", "镜头")),
        ("short_duration", ("出现时间短",)),
    ]
    flags = []
    for name, keywords in checks:
        if has(note, *keywords):
            flags.append(name)
    return flags


def human_label_from_note(note):
    """Map the user's detailed natural-language note to a standard keep/reason."""
    bucket = annotation_bucket(note)
    flags = set(detailed_note_flags(note))

    if bucket == "unable_open":
        return "no", "unable_open"
    if bucket == "cannot_see_suspect_vehicle":
        return "no", "cannot_see_target"
    if bucket == "collision_process_clear":
        return "yes", "collision_visible"
    if bucket == "collision_process_weak":
        return "yes", "collision_weak"
    if bucket == "no_process_but_damage_trace":
        return "yes", "damage_trace"

    if "no_evidence" in flags and not ({"person_handling", "explicit_evidence"} & flags):
        return "no", "weak_evidence"

    if "single_vehicle" in flags:
        if {"person_handling", "explicit_evidence"} & flags:
            return "uncertain", "single_vehicle_but_evidence"
        return "no", "single_vehicle_parking"

    if bucket == "no_process_parking_no_person":
        if {"person_handling", "explicit_evidence"} & flags:
            return "yes", "post_incident_handling"
        return "no", "parking_no_person_no_evidence"

    if bucket == "no_process_parking_with_person":
        if "explicit_evidence" in flags:
            return "yes", "post_incident_handling_with_evidence"
        if {"two_or_more_close", "person_handling"} <= flags:
            if {"low_quality", "small_or_edge", "occluded"} & flags:
                return "uncertain", "post_incident_handling_visual_weak"
            return "yes", "post_incident_handling"
        if {"far_or_unclear_relation", "person_handling"} <= flags:
            return "uncertain", "far_but_handling"
        if "person_handling" in flags:
            return "uncertain", "handling_relation_unclear"
        return "uncertain", "parking_with_person_unspecified"

    return "uncertain", "other"


def text_features(text):
    positive = [
        "碰撞",
        "追尾",
        "刮擦",
        "剐蹭",
        "相撞",
        "撞击",
        "人员下车",
        "下车查看",
        "下车交涉",
        "车门打开",
        "双闪",
        "警示牌",
        "斜停",
        "横跨",
    ]
    negative = [
        "未见明显事故",
        "未见事故",
        "无事故",
        "未见明显碰撞",
        "无明显碰撞",
        "无碰撞",
        "未见人员下车",
        "无人员下车",
        "正常行驶",
    ]
    return {
        "text_positive_hits": "|".join([kw for kw in positive if kw in text]),
        "text_negative_hits": "|".join([kw for kw in negative if kw in text]),
    }


def default_lcrm_args():
    # Defaults mirror postprocess_lcrm_lite_fusion.py. The review CSV uses these
    # only as auxiliary evidence, not as final labels.
    return SimpleNamespace(
        far_y=0.45,
        mid_y=0.75,
        far_weight=0.45,
        mid_weight=0.80,
        min_area_ratio=0.0015,
        max_pair_area_ratio=0.0,
        close_norm_gap=0.35,
        lateral_suppress_dx=0.40,
        lateral_suppress_dy=0.20,
        min_pair_compactness=0.35,
        min_track_frames=3,
        min_static_count=4,
        score_thr_weak=0.75,
        max_norm_gap_weak=0.10,
        min_persistence_weak=4,
        construction_bypass_score=0.0,
        construction_bypass_score_congestion=None,
        construction_bypass_static_count=3,
        stationary_step=0.04,
        persistence_norm=4.0,
        close_weight=0.40,
        persistence_weight=0.35,
        stationary_weight=0.25,
        score_thr=0.65,
        congestion_vehicle_peak=12,
        congestion_slow_ratio=0.70,
        congestion_slow_regions=3,
        congestion_score_penalty=0.10,
        congestion_min_compactness=0.70,
        congestion_min_persistence=8,
        congestion_max_static=5,
        congestion_band_tolerance=0.15,
        congestion_min_relative_compactness=1.75,
        motorcycle_peak=3,
        twowheel_exempt_static_count=4,
        twowheel_exempt_local_pair_raw=0.45,
        twowheel_exempt_contact_frames=3,
        twowheel_exempt_close_frames=8,
        enable_area_relaxed_retry=False,
        area_relaxed_ratio_mul=0.5,
        area_relaxed_min_area_ratio=0.0004,
        area_relaxed_min_contact_frames=3,
        area_relaxed_min_close_frames=8,
        person_bonus=0.0,
        person_bonus_min_peak=3,
        congestion_strong_rel_compact=99.0,
        congestion_strong_min_sc=0,
        congestion_strong_max_compact=1.0,
        cone_peak=8,
        construction_vehicle_peak=4,
        cone_pair_peak=5,
        barrier_pair_peak=3,
        block_debris_context=False,
        debris_peak=1,
        road_obstacle_peak=3,
    )


def lcrm_features(ovd_row, args):
    empty = {
        "ovd_found": False,
        "lcrm_ok": "",
        "lcrm_signal": "",
        "lcrm_reason": "",
        "lcrm_channel": "",
        "lcrm_score": "",
        "lcrm_compactness": "",
        "lcrm_dx_union": "",
        "lcrm_dy_union": "",
        "lcrm_edge_norm_gap": "",
        "lcrm_persistence": "",
        "lcrm_static_count": "",
        "lcrm_person_peak": "",
        "lcrm_vehicle_peak": "",
    }
    ovd = get_ovd_payload(ovd_row or {})
    if not ovd:
        return empty
    lcrm = ovd.get("lcrm") if isinstance(ovd, dict) else None
    if not isinstance(lcrm, dict):
        empty["ovd_found"] = True
        return empty
    signal, debug = lcrm_lite_signal(lcrm, args)
    best = debug.get("best_pair") or {}
    ctx = lcrm.get("semantic_context") or {}
    return {
        "ovd_found": True,
        "lcrm_ok": bool(lcrm.get("ok")),
        "lcrm_signal": bool(signal),
        "lcrm_reason": debug.get("reason", ""),
        "lcrm_channel": debug.get("channel", ""),
        "lcrm_score": best.get("score", ""),
        "lcrm_compactness": best.get("compactness", ""),
        "lcrm_dx_union": best.get("dx_union", ""),
        "lcrm_dy_union": best.get("dy_union", ""),
        "lcrm_edge_norm_gap": best.get("edge_norm_gap", ""),
        "lcrm_persistence": best.get("persistence_frames", ""),
        "lcrm_static_count": debug.get("static_count", lcrm.get("static_count", "")),
        "lcrm_person_peak": ctx.get("person_peak", best.get("person_peak", "")),
        "lcrm_vehicle_peak": lcrm.get("vehicle_peak", ""),
    }


def suggest(row):
    bucket = row["annotation_bucket"]
    flags = set(row["note_flags"].split("|")) if row["note_flags"] else set()
    base_label = row.get("base_label") or ""
    positive_hits = row.get("text_positive_hits") or ""
    lcrm_signal = str(row.get("lcrm_signal", "")).lower() == "true"

    if bucket in {"unable_open", "cannot_see_suspect_vehicle"}:
        return "drop", "invalid_or_unverifiable"
    if bucket in {"collision_process_clear", "collision_process_weak", "no_process_but_damage_trace"}:
        return "keep", bucket
    if bucket == "no_process_parking_no_person":
        if "damage_trace" in flags or lcrm_signal:
            return "review", "no_person_but_some_accident_evidence"
        return "drop", "parking_without_handling_evidence"
    if bucket == "no_process_parking_with_person":
        weak_flags = flags & {"low_quality", "edge_or_small", "occluded", "far_apart"}
        if weak_flags:
            return "review", "post_incident_handling_but_visual_weak"
        if base_label == "多车事故" or positive_hits:
            return "keep", "post_incident_handling"
        return "review", "post_incident_handling_needs_confirmation"
    return "review", "unclassified"


def main():
    ap = argparse.ArgumentParser(
        description="Build a CSV for manual review of multicar accident evidence labels."
    )
    ap.add_argument("--manual-txt", required=True, help="Your manually annotated 多车事故.txt.")
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--base-pred", help="Optional E8/base prediction JSONL.")
    ap.add_argument("--base-pred-field", default="pred")
    ap.add_argument("--ovd", help="Optional OVD/LCRM JSONL.")
    ap.add_argument("--gt-field", default="gt")
    ap.add_argument(
        "--fill-human-from-note",
        action="store_true",
        help="Auto-fill human_keep/human_reason from detailed natural-language manual notes.",
    )
    args = ap.parse_args()

    manual_rows = parse_manual_txt(args.manual_txt)
    base_map = by_event_id(read_jsonl(args.base_pred)) if args.base_pred else {}
    ovd_map = by_event_id(read_jsonl(args.ovd)) if args.ovd else {}
    lcrm_args = default_lcrm_args()

    out_rows = []
    bucket_counts = Counter()
    suggest_counts = Counter()
    for manual in manual_rows:
        event_id = manual["event_id"]
        base_row = base_map.get(event_id, {})
        ovd_row = ovd_map.get(event_id, {})
        pred_text = pick_text(base_row, args.base_pred_field)
        base_label = detect_label(pred_text) or UNKNOWN_LABEL if pred_text else ""
        video = base_row.get("video") or ovd_row.get("video") or ""
        gt_label = get_gt_label(base_row, args.gt_field) if base_row else ""
        bucket = annotation_bucket(manual["manual_note"])
        flags = note_flags(manual["manual_note"])

        rec = {
            **manual,
            "video": video,
            "gt_label": gt_label,
            "annotation_bucket": bucket,
            "note_flags": "|".join(sorted(set(flags + detailed_note_flags(manual["manual_note"])))),
            "base_label": base_label,
            "base_pred_excerpt": brief(pred_text),
            **text_features(pred_text),
            **lcrm_features(ovd_row, lcrm_args),
            "human_keep": "",
            "human_reason": "",
            "human_note": "",
        }
        auto_suggest, auto_reason = suggest(rec)
        rec["auto_suggest"] = auto_suggest
        rec["auto_reason"] = auto_reason
        if args.fill_human_from_note:
            rec["human_keep"], rec["human_reason"] = human_label_from_note(manual["manual_note"])
        out_rows.append(rec)
        bucket_counts[bucket] += 1
        suggest_counts[auto_suggest] += 1

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "idx",
        "event_id",
        "video",
        "gt_label",
        "manual_note",
        "annotation_bucket",
        "note_flags",
        "auto_suggest",
        "auto_reason",
        "human_keep",
        "human_reason",
        "human_note",
        "base_label",
        "base_pred_excerpt",
        "text_positive_hits",
        "text_negative_hits",
        "ovd_found",
        "lcrm_ok",
        "lcrm_signal",
        "lcrm_reason",
        "lcrm_channel",
        "lcrm_score",
        "lcrm_compactness",
        "lcrm_dx_union",
        "lcrm_dy_union",
        "lcrm_edge_norm_gap",
        "lcrm_persistence",
        "lcrm_static_count",
        "lcrm_person_peak",
        "lcrm_vehicle_peak",
    ]
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print("output:", out)
    print("total:", len(out_rows))
    print("annotation_bucket:", dict(bucket_counts))
    print("auto_suggest:", dict(suggest_counts))
    if args.fill_human_from_note:
        print("human_keep:", dict(Counter(row["human_keep"] for row in out_rows)))
        print("human_reason:", dict(Counter(row["human_reason"] for row in out_rows)))


if __name__ == "__main__":
    main()
