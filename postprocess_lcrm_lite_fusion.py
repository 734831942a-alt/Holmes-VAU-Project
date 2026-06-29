import argparse
import json
import math
from collections import Counter
from pathlib import Path

from eval_traffic_fixed import LABELS as FIXED_EVAL_LABELS
from eval_traffic_fixed import detect_label as fixed_detect_label


MULTICAR_LABEL = "\u591a\u8f66\u4e8b\u6545"
CONGESTION_LABEL = "\u62e5\u5835"
PARKING_LABEL = "\u5f02\u5e38\u505c\u8f66"
CONSTRUCTION_LABEL = "\u5360\u9053\u65bd\u5de5"
TWOWHEEL_LABEL = "\u4e8c\u8f6e\u8f66\u8f86\u95ef\u5165"
DEBRIS_LABEL = "\u629b\u6d12\u7269"

DEFAULT_LABELS = list(FIXED_EVAL_LABELS)


def parse_csv(text):
    return [x.strip() for x in str(text or "").split(",") if x.strip()]


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
    return {row.get("video"): row for row in rows if row.get("video")}


def label_from_video(video):
    return str(video or "").split("_")[0]


def detect_label(text):
    return fixed_detect_label(text)


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
    if any(k in row for k in ["lcrm", "vscm", "sarp", "peak_count"]):
        return row
    return {}


def box_center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_diag(box):
    return math.hypot(max(0.0, box[2] - box[0]), max(0.0, box[3] - box[1]))


def edge_gap(a, b):
    x_gap = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    y_gap = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.hypot(x_gap, y_gap)


def pair_union_geometry(a, b):
    ca = box_center(a)
    cb = box_center(b)
    union_w = max(1.0, max(a[2], b[2]) - min(a[0], b[0]))
    union_h = max(1.0, max(a[3], b[3]) - min(a[1], b[1]))
    dx_union = abs(ca[0] - cb[0]) / union_w
    dy_union = abs(ca[1] - cb[1]) / union_h
    compactness = 1.0 - math.hypot(dx_union, dy_union) / math.sqrt(2.0)
    return {
        "dx_union": max(0.0, min(1.0, dx_union)),
        "dy_union": max(0.0, min(1.0, dy_union)),
        "compactness": max(0.0, min(1.0, compactness)),
    }


def infer_image_size(tracks):
    widths = []
    heights = []
    for tr in tracks:
        box = tr.get("last_box")
        center = tr.get("last_center")
        if not box or not center or len(center) < 2:
            continue
        cx_abs, cy_abs = box_center(box)
        if center[0] > 1e-6:
            widths.append(cx_abs / center[0])
        if center[1] > 1e-6:
            heights.append(cy_abs / center[1])
    if not widths or not heights:
        return None, None
    widths.sort()
    heights.sort()
    return widths[len(widths) // 2], heights[len(heights) // 2]


def depth_weight(pair_cy, args):
    if pair_cy < args.far_y:
        return args.far_weight
    if pair_cy < args.mid_y:
        return args.mid_weight
    return 1.0


def semantic_context(lcrm):
    ctx = lcrm.get("semantic_context") if isinstance(lcrm, dict) else {}
    return ctx if isinstance(ctx, dict) else {}


def blocked_by_protected_context(lcrm, args):
    ctx = semantic_context(lcrm)
    motorcycle_peak = int(ctx.get("motorcycle_peak", 0) or 0)
    cone_peak = int(ctx.get("traffic_cone_peak", 0) or 0)
    construction_vehicle_peak = int(ctx.get("construction_vehicle_peak", 0) or 0)
    road_barrier_peak = int(ctx.get("road_barrier_peak", 0) or 0)
    debris_peak = int(ctx.get("debris_peak", 0) or 0)
    road_obstacle_peak = int(ctx.get("road_obstacle_peak", 0) or 0)

    static_count = int(lcrm.get("static_count", 0) or 0)
    local_pair_raw_score = float(lcrm.get("local_pair_raw_score", 0.0) or 0.0)
    max_pair_contact_frames = int(lcrm.get("max_pair_contact_frames", 0) or 0)
    max_pair_close_frames = int(lcrm.get("max_pair_close_frames", 0) or 0)

    # Twowheel suppression is now conditional: when accident evidence is already
    # strong (enough static pair/contact evidence), do not let sparse motorcycle
    # detections hard-veto the sample.
    if motorcycle_peak >= args.motorcycle_peak:
        accident_evidence_strong = (
            static_count >= args.twowheel_exempt_static_count
            or local_pair_raw_score >= args.twowheel_exempt_local_pair_raw
            or max_pair_contact_frames >= args.twowheel_exempt_contact_frames
            or max_pair_close_frames >= args.twowheel_exempt_close_frames
        )
        if not accident_evidence_strong:
            return True, "twowheel_context"
    construction = (
        cone_peak >= args.cone_peak
        or construction_vehicle_peak >= args.construction_vehicle_peak
        or (cone_peak >= args.cone_pair_peak and road_barrier_peak >= args.barrier_pair_peak)
    )
    if construction:
        return True, "construction_context"
    if args.block_debris_context and (debris_peak >= args.debris_peak or road_obstacle_peak >= args.road_obstacle_peak):
        return True, "debris_context"
    return False, None


def text_evidence_count(text, keywords):
    text = "" if text is None else str(text)
    return sum(1 for kw in keywords if kw and kw in text)


TEXT_POSITIVE_ACCIDENT = [
    "碰撞",
    "追尾",
    "刮擦",
    "剐蹭",
    "相撞",
    "撞击",
    "连环",
]

TEXT_POSITIVE_SCENE = [
    "人员下车",
    "驾驶员下车",
    "下车查看",
    "下车交涉",
    "车门打开",
    "开启双闪",
    "双闪",
    "警示牌",
    "三角警示牌",
    "斜停",
    "横跨",
]

TEXT_NEGATIVE_ACCIDENT = [
    "未见明显事故",
    "未见事故",
    "无事故",
    "无明显事故",
    "未见明显碰撞",
    "未见碰撞",
    "无明显碰撞",
    "无碰撞",
    "未见碰撞痕迹",
    "无碰撞痕迹",
    "未见明显人员下车",
    "未见人员下车",
    "无人员下车",
    "正常行驶",
]


def text_semantic_features(text):
    text = "" if text is None else str(text)
    cleaned = text
    neg_hits = []
    for kw in TEXT_NEGATIVE_ACCIDENT:
        if kw in cleaned:
            neg_hits.append(kw)
            cleaned = cleaned.replace(kw, "")
    accident_hits = [kw for kw in TEXT_POSITIVE_ACCIDENT if kw in cleaned]
    scene_hits = [kw for kw in TEXT_POSITIVE_SCENE if kw in cleaned]
    return {
        "positive_accident": len(accident_hits),
        "positive_scene": len(scene_hits),
        "negative_accident": len(neg_hits),
        "positive_hits": accident_hits + scene_hits,
        "negative_hits": neg_hits,
    }


def apply_semantic_text_gate(signal, debug, lcrm, args, base_text):
    """Optional low-cost semantic gate using existing model text and OVD context."""
    mode = args.text_semantic_mode
    if not signal or mode == "off":
        return signal, debug

    features = text_semantic_features(base_text)
    ctx = semantic_context(lcrm)
    person_peak = int(ctx.get("person_peak", 0) or 0) if isinstance(ctx, dict) else 0
    contact_frames = int(lcrm.get("max_pair_contact_frames", 0) or 0) if isinstance(lcrm, dict) else 0
    close_frames = int(lcrm.get("max_pair_close_frames", 0) or 0) if isinstance(lcrm, dict) else 0
    best = debug.get("best_pair") or {}

    positive_count = features["positive_accident"] + features["positive_scene"]
    text_confirmed = positive_count >= args.text_positive_min
    context_confirmed = (
        person_peak >= args.semantic_person_min_peak
        or contact_frames >= args.semantic_contact_min_frames
        or close_frames >= args.semantic_close_min_frames
    )
    geometry_confirmed = (
        float(best.get("score", 0.0) or 0.0) >= args.semantic_confirm_score
        and float(best.get("compactness", 0.0) or 0.0) >= args.semantic_confirm_compactness
        and int(best.get("persistence_frames", 0) or 0) >= args.semantic_confirm_persistence
    )
    neg_veto = features["negative_accident"] >= args.text_neg_veto_min

    semantic_debug = {
        **features,
        "mode": mode,
        "person_peak": person_peak,
        "contact_frames": contact_frames,
        "close_frames": close_frames,
        "text_confirmed": bool(text_confirmed),
        "context_confirmed": bool(context_confirmed),
        "geometry_confirmed": bool(geometry_confirmed),
        "negative_veto": bool(neg_veto),
    }
    debug["semantic_text"] = semantic_debug

    blocked = False
    reason = None
    if mode == "veto":
        blocked = neg_veto and not (text_confirmed or context_confirmed)
        reason = "semantic_text_veto"
    elif mode == "confirm":
        blocked = not (text_confirmed or context_confirmed)
        reason = "semantic_text_unconfirmed"
    elif mode == "balanced":
        blocked = (
            (neg_veto and not (text_confirmed or context_confirmed))
            or not (text_confirmed or context_confirmed or geometry_confirmed)
        )
        reason = "semantic_text_balanced_block"

    if blocked:
        debug["pre_semantic_reason"] = debug.get("reason")
        debug["reason"] = reason
        debug["channel"] = None
        return False, debug
    return signal, debug


def lcrm_lite_signal(lcrm, args, base_label=None, base_text=""):
    if not isinstance(lcrm, dict):
        return False, {"reason": "lcrm_missing"}
    if not lcrm.get("ok"):
        return False, {"reason": "lcrm_not_ok", "lcrm_reason": lcrm.get("reason")}

    blocked, block_reason = blocked_by_protected_context(lcrm, args)
    # construction_context may be bypassed after pair scoring; others are hard veto.
    construction_blocked = blocked and block_reason == "construction_context"
    if blocked and not construction_blocked:
        return False, {"reason": block_reason}

    tracks = [tr for tr in lcrm.get("tracks", []) if isinstance(tr, dict)]
    img_w, img_h = infer_image_size(tracks)
    if not img_w or not img_h:
        return False, {"reason": "missing_image_size"}

    static_tracks = [
        tr for tr in tracks
        if tr.get("bucket") == "static"
        and int(tr.get("frames", 0) or 0) >= args.min_track_frames
        and float(tr.get("max_step_norm", 1.0) or 1.0) <= args.stationary_step
        and tr.get("last_box")
    ]
    static_count = int(lcrm.get("static_count", len(static_tracks)) or 0)
    # Minimum 2 static tracks required to form any pair.
    if static_count < 2:
        return False, {
            "reason": "static_count_below_threshold",
            "static_count": static_count,
            "required_static_count": args.min_static_count,
            "static_tracks": len(static_tracks),
            "track_count": lcrm.get("track_count"),
        }
    if len(static_tracks) < 2:
        return False, {
            "reason": "not_enough_static_tracks",
            "static_tracks": len(static_tracks),
            "track_count": lcrm.get("track_count"),
        }

    best = None

    def _find_best_pair(min_area_ratio):
        best_item = None
        band_meta = []
        for i in range(len(static_tracks)):
            for j in range(i + 1, len(static_tracks)):
                a = static_tracks[i]["last_box"]
                b = static_tracks[j]["last_box"]
                ca = box_center(a)
                cb = box_center(b)
                pair_cy = ((ca[1] + cb[1]) / 2.0) / img_h
                max_area_ratio = max(box_area(a), box_area(b)) / max(1.0, img_w * img_h)
                if max_area_ratio < min_area_ratio:
                    continue
                if args.max_pair_area_ratio > 0 and max_area_ratio > args.max_pair_area_ratio:
                    continue
                gap = edge_gap(a, b)
                max_diag = max(1.0, box_diag(a), box_diag(b))
                edge_norm_gap = gap / max_diag
                geom = pair_union_geometry(a, b)
                # Lateral side-by-side suppression:
                # If the pair's x-centre distance is a large fraction of the union width
                # (vehicles clearly in adjacent lanes) while y-distance is small
                # (same depth), the geometry is side-by-side, not a collision.
                # Typical false-positive: two large trucks driving abreast.
                if (
                    args.lateral_suppress_dx > 0
                    and geom["dx_union"] > args.lateral_suppress_dx
                    and geom["dy_union"] < args.lateral_suppress_dy
                ):
                    continue
                # Minimum compactness gate:
                # Very low compactness means the two vehicles are far apart in the
                # frame (diagonal or opposite corners) — not a plausible collision pair.
                # Typical false-positive: one stopped car on shoulder + unrelated
                # slow vehicle elsewhere in the scene.
                if (
                    args.min_pair_compactness > 0
                    and geom["compactness"] < args.min_pair_compactness
                ):
                    continue
                close_score = geom["compactness"]
                persistence_frames = min(int(static_tracks[i].get("frames", 0) or 0), int(static_tracks[j].get("frames", 0) or 0))
                persistence_score = min(1.0, persistence_frames / max(1.0, args.persistence_norm))
                stationary_score = 1.0
                d_weight = depth_weight(pair_cy, args)
                score = (
                    args.close_weight * close_score
                    + args.persistence_weight * persistence_score
                    + args.stationary_weight * stationary_score
                ) * d_weight
                item = {
                    "score": round(score, 4),
                    "norm_gap": round(1.0 - close_score, 4),
                    "edge_norm_gap": round(edge_norm_gap, 4),
                    "dx_union": round(geom["dx_union"], 4),
                    "dy_union": round(geom["dy_union"], 4),
                    "compactness": round(geom["compactness"], 4),
                    "close_score": round(close_score, 4),
                    "persistence_frames": persistence_frames,
                    "persistence_score": round(persistence_score, 4),
                    "pair_cy": round(pair_cy, 4),
                    "depth_weight": d_weight,
                    "max_area_ratio": round(max_area_ratio, 6),
                    "track_ids": [static_tracks[i].get("track_id"), static_tracks[j].get("track_id")],
                }
                band_meta.append((pair_cy, geom["compactness"]))
                if best_item is None or item["score"] > best_item["score"]:
                    best_item = item
        return best_item, band_meta

    # All valid pairs' (pair_cy, compactness) for relative-compactness computation.
    _band_meta: list[tuple[float, float]] = []
    best, _band_meta = _find_best_pair(args.min_area_ratio)
    area_relaxed_retry = False
    area_retry_min_area_ratio = args.min_area_ratio

    if best is None and args.enable_area_relaxed_retry:
        max_pair_contact_frames = int(lcrm.get("max_pair_contact_frames", 0) or 0)
        max_pair_close_frames = int(lcrm.get("max_pair_close_frames", 0) or 0)
        if (
            max_pair_contact_frames >= args.area_relaxed_min_contact_frames
            or max_pair_close_frames >= args.area_relaxed_min_close_frames
        ):
            area_retry_min_area_ratio = max(args.area_relaxed_min_area_ratio, args.min_area_ratio * args.area_relaxed_ratio_mul)
            if area_retry_min_area_ratio < args.min_area_ratio:
                best, _band_meta = _find_best_pair(area_retry_min_area_ratio)
                area_relaxed_retry = best is not None

    if best is None:
        return False, {
            "reason": "no_pair_after_area_filter",
            "static_tracks": len(static_tracks),
            "static_count": static_count,
            "min_area_ratio": args.min_area_ratio,
            "max_pair_area_ratio": args.max_pair_area_ratio,
            "area_relaxed_retry": bool(area_relaxed_retry),
            "area_retry_min_area_ratio": area_retry_min_area_ratio,
        }

    # Relative compactness: how much more compact is the best pair compared to
    # other static pairs in the same depth band?
    # high relative_compactness → pair is anomalously tight (accident evidence)
    # relative_compactness ≈ 1.0 → pair is just as compact as background (congestion)
    _band_tol = args.congestion_band_tolerance
    _same_band = [c for (cy, c) in _band_meta if abs(cy - best["pair_cy"]) <= _band_tol]
    if len(_same_band) >= 2:
        _same_band_sorted = sorted(_same_band)
        _median_band = _same_band_sorted[len(_same_band_sorted) // 2]
        _rel_compact = best["compactness"] / max(0.01, _median_band)
    else:
        _median_band = None
        _rel_compact = None
    best["relative_compactness"] = round(_rel_compact, 3) if _rel_compact is not None else None
    best["median_band_compactness"] = round(_median_band, 3) if _median_band is not None else None
    best["band_pair_count"] = len(_same_band)

    # Person-evidence score bonus: bystanders/responders at accident scene boost confidence.
    _ctx = semantic_context(lcrm)
    _person_peak = int(_ctx.get("person_peak", 0) or 0)
    _person_bonus_applied = 0.0
    if args.person_bonus > 0 and _person_peak >= args.person_bonus_min_peak:
        _person_bonus_applied = args.person_bonus
        best["score"] = round(best["score"] + _person_bonus_applied, 4)
    best["person_peak"] = _person_peak
    best["person_bonus"] = round(_person_bonus_applied, 4)

    required = args.score_thr
    congestion_like = (
        int(lcrm.get("vehicle_peak", 0) or 0) >= args.congestion_vehicle_peak
        and float(lcrm.get("global_slow_ratio", 0.0) or 0.0) >= args.congestion_slow_ratio
        and int(lcrm.get("slow_region_count", 0) or 0) >= args.congestion_slow_regions
    )
    if congestion_like:
        required += args.congestion_score_penalty

    # Dual-channel logic:
    # Strong channel: enough global static vehicles + score meets base threshold.
    strong_channel = static_count >= args.min_static_count and best["score"] >= required
    # Weak channel: fewer static vehicles but the pair itself must be very high quality.
    # Disabled when congestion_like (拥堵背景下弱通道不允许触发).
    weak_channel = (
        args.score_thr_weak > 0
        and not congestion_like
        and best["score"] >= args.score_thr_weak
        and best["norm_gap"] <= args.max_norm_gap_weak
        and best["persistence_frames"] >= args.min_persistence_weak
    )

    # Congestion-accident channel: fires only when the scene is congestion-like
    # but a very compact, long-persisting static pair stands out (crashed vehicles).
    # Uses both absolute compactness AND relative compactness vs. same-depth-band peers.
    # relative_compactness: best pair / median of other pairs in same depth band.
    #   ≫ 1 → this pair is anomalously tight compared to background → accident evidence.
    #   ≈ 1 → pair is just as compact as background → likely just congestion.
    # When fewer than 2 pairs exist in the band, fall back to absolute compactness only.
    _rel = best.get("relative_compactness")
    _rel_ok = (
        _rel is None  # only pair in band → no background, use absolute only
        or _rel >= args.congestion_min_relative_compactness
    )
    congestion_accident_channel = (
        args.congestion_min_compactness > 0
        and congestion_like
        and static_count <= args.congestion_max_static
        and best["compactness"] >= args.congestion_min_compactness
        and _rel_ok
        and best["persistence_frames"] >= args.congestion_min_persistence
    )

    # Construction bypass: only if explicitly enabled and both score and static_count
    # clear a higher bar. Only the strong channel qualifies for bypass.
    construction_bypass = False
    if construction_blocked:
        bypass = (
            args.construction_bypass_score > 0
            and static_count >= args.construction_bypass_static_count
            and best["score"] >= args.construction_bypass_score
        )
        if not bypass:
            return False, {
                "reason": "construction_context",
                "best_pair": best,
                "static_count": static_count,
            }
        construction_bypass = True
        # Only strong channel may survive construction bypass.
        weak_channel = False
        congestion_accident_channel = False

    signal = strong_channel or weak_channel
    channel = "strong" if strong_channel else ("weak" if weak_channel else None)
    debug = {
        "reason": "lcrm_lite_core" if signal else "score_below_threshold",
        "channel": channel,
        "required_score": round(required, 4),
        "congestion_like": bool(congestion_like),
        "best_pair": best,
        "static_count": static_count,
        "required_static_count": args.min_static_count,
        "slow_count": lcrm.get("slow_count"),
        "moving_count": lcrm.get("moving_count"),
        "global_slow_ratio": lcrm.get("global_slow_ratio"),
        "vehicle_peak": lcrm.get("vehicle_peak"),
        "slow_region_count": lcrm.get("slow_region_count"),
        "primary_signal": lcrm.get("primary_signal"),
        "local_pair_score": lcrm.get("local_pair_score"),
        "congestion_accident_channel": bool(congestion_accident_channel),
        "construction_bypass": bool(construction_bypass),
        "area_relaxed_retry": bool(area_relaxed_retry),
        "area_retry_min_area_ratio": area_retry_min_area_ratio,
    }
    return signal, debug


def vscm_debug_lite_signal(vscm_debug, args):
    """Fallback for old OVD summaries that have vscm_debug but no lcrm tracks."""
    if not isinstance(vscm_debug, dict):
        return False, {"reason": "vscm_debug_missing"}

    static_count = int(vscm_debug.get("static_count", 0) or 0)
    moving_count = int(vscm_debug.get("moving_count", 0) or 0)
    close_frames = int(vscm_debug.get("close_frames", 0) or 0)
    contact_frames = int(vscm_debug.get("contact_frames", 0) or 0)
    stagnation_frames = int(vscm_debug.get("pair_stagnation_frames", 0) or 0)
    queue_like = bool(vscm_debug.get("queue_like"))
    proximity = bool(vscm_debug.get("proximity"))
    proximity_reason = vscm_debug.get("proximity_reason")
    context_hits = vscm_debug.get("context_hits") or {}

    person_hits = int(context_hits.get("person", 0) or 0)
    cone_hits = int(context_hits.get("traffic cone", 0) or 0)
    construction_hits = int(context_hits.get("construction vehicle", 0) or 0)
    barrier_hits = int(context_hits.get("road barrier", 0) or 0)

    if cone_hits >= args.cone_peak or construction_hits >= args.construction_vehicle_peak:
        return False, {
            "reason": "construction_context",
            "static_count": static_count,
            "close_frames": close_frames,
            "contact_frames": contact_frames,
            "pair_stagnation_frames": stagnation_frames,
            "context_hits": context_hits,
        }

    relation_score = min(1.0, max(close_frames, contact_frames, stagnation_frames) / max(1.0, args.vscm_relation_norm))
    static_score = 1.0 if 2 <= static_count <= 4 else (0.5 if static_count in {5, 6} else 0.0)
    context_score = min(1.0, (person_hits + 0.5 * barrier_hits) / max(1.0, args.vscm_context_norm))
    queue_penalty = args.vscm_queue_penalty if queue_like else 0.0
    score = 0.45 * relation_score + 0.35 * static_score + 0.20 * context_score - queue_penalty

    signal = (
        score >= args.vscm_debug_score_thr
        and static_count >= 2
        and (proximity or stagnation_frames >= args.vscm_min_stagnation_frames or contact_frames >= args.vscm_min_contact_frames)
    )
    return signal, {
        "reason": "vscm_debug_lite_core" if signal else "vscm_debug_score_below_threshold",
        "score": round(score, 4),
        "relation_score": round(relation_score, 4),
        "static_score": round(static_score, 4),
        "context_score": round(context_score, 4),
        "queue_like": queue_like,
        "queue_penalty": queue_penalty,
        "static_count": static_count,
        "moving_count": moving_count,
        "close_frames": close_frames,
        "contact_frames": contact_frames,
        "pair_stagnation_frames": stagnation_frames,
        "proximity": proximity,
        "proximity_reason": proximity_reason,
        "context_hits": context_hits,
    }


def main():
    ap = argparse.ArgumentParser(
        description="CPU-only LCRM-lite fusion using stored OVD lcrm tracks; no GPU inference."
    )
    ap.add_argument("--base-pred", required=True)
    ap.add_argument("--ovd", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--base-pred-field", default="pred")
    ap.add_argument("--fused-field", default="pred_lcrm_lite_fused")
    ap.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    ap.add_argument("--allowed-base-labels", default=PARKING_LABEL, type=parse_csv)
    ap.add_argument(
        "--construction-bypass-extra-base-labels",
        default="",
        type=parse_csv,
        help="Extra base labels allowed to flip only when construction_bypass is true.",
    )
    ap.add_argument(
        "--source",
        choices=["auto", "lcrm", "vscm_debug"],
        default="auto",
        help="auto uses lcrm when present, otherwise falls back to vscm_debug.",
    )
    ap.add_argument("--far-y", type=float, default=0.45)
    ap.add_argument("--mid-y", type=float, default=0.75)
    ap.add_argument("--far-weight", type=float, default=0.45)
    ap.add_argument("--mid-weight", type=float, default=0.80)
    ap.add_argument("--min-area-ratio", type=float, default=0.0015)
    ap.add_argument(
        "--max-pair-area-ratio",
        type=float,
        default=0.0,
        help="Optional upper bound for the larger box area ratio in a static pair. <=0 disables it.",
    )
    ap.add_argument("--close-norm-gap", type=float, default=0.35)
    ap.add_argument("--lateral-suppress-dx", type=float, default=0.40,
                        help="Skip a static pair when dx_union > this AND dy_union < "
                             "--lateral-suppress-dy: the vehicles are side-by-side in "
                             "adjacent lanes, not a collision geometry. 0 = disabled.")
    ap.add_argument("--lateral-suppress-dy", type=float, default=0.20,
                        help="Upper dy_union bound for lateral suppression (see --lateral-suppress-dx).")
    ap.add_argument("--min-pair-compactness", type=float, default=0.35,
                        help="Skip a static pair whose compactness is below this threshold. "
                             "Low compactness means the two vehicles are far apart in the frame "
                             "(diagonal or opposite corners) and are unlikely to have collided. "
                             "0 = disabled.")
    ap.add_argument("--min-track-frames", type=int, default=3)
    ap.add_argument("--min-static-count", type=int, default=4,
                        help="Strong channel: global static_count must reach this.")
    ap.add_argument("--score-thr-weak", type=float, default=0.75,
                        help="Weak channel score threshold (static_count>=2). 0 disables weak channel.")
    ap.add_argument("--max-norm-gap-weak", type=float, default=0.10,
                        help="Weak channel: best_pair.norm_gap must be <= this.")
    ap.add_argument("--min-persistence-weak", type=int, default=4,
                        help="Weak channel: best_pair.persistence_frames must be >= this.")
    ap.add_argument("--construction-bypass-score", type=float, default=0.0,
                        help="If >0, strong channel may bypass construction_context when score >= this. 0=disabled.")
    ap.add_argument("--construction-bypass-score-congestion", type=float, default=None,
                        help="For 拥堵 base: construction bypass additionally requires score >= this. "
                             "Defaults to --construction-bypass-score if unset (backward-compatible).")
    ap.add_argument("--construction-bypass-static-count", type=int, default=3,
                        help="Min static_count needed for construction bypass.")
    ap.add_argument("--stationary-step", type=float, default=0.04)
    ap.add_argument("--persistence-norm", type=float, default=4.0)
    ap.add_argument("--close-weight", type=float, default=0.40)
    ap.add_argument("--persistence-weight", type=float, default=0.35)
    ap.add_argument("--stationary-weight", type=float, default=0.25)
    ap.add_argument("--score-thr", type=float, default=0.65)
    ap.add_argument("--congestion-vehicle-peak", type=int, default=12)
    ap.add_argument("--congestion-slow-ratio", type=float, default=0.70)
    ap.add_argument("--congestion-slow-regions", type=int, default=3)
    ap.add_argument("--congestion-score-penalty", type=float, default=0.10)
    # Congestion-accident channel (base=拥堵 → 多车事故): stricter geometric criteria.
    # Set --congestion-min-compactness 0 to disable this channel entirely.
    ap.add_argument("--congestion-min-compactness", type=float, default=0.70,
                        help="Congestion-accident channel: best pair compactness must reach this. 0=disabled.")
    ap.add_argument("--congestion-min-persistence", type=int, default=8,
                        help="Congestion-accident channel: best pair persistence_frames must reach this.")
    ap.add_argument("--congestion-max-static", type=int, default=5,
                        help="Congestion-accident channel: global static_count must NOT exceed this (few specifically stopped cars).")
    ap.add_argument("--congestion-band-tolerance", type=float, default=0.15,
                        help="Depth-band half-width for relative-compactness computation (in normalised cy units).")
    ap.add_argument("--congestion-min-relative-compactness", type=float, default=1.75,
                        help="Congestion-accident channel: best pair must be >= this × median band compactness.")
    ap.add_argument("--motorcycle-peak", type=int, default=3)
    ap.add_argument("--twowheel-exempt-static-count", type=int, default=4,
                        help="If static_count >= this, motorcycle_context will not hard-veto.")
    ap.add_argument("--twowheel-exempt-local-pair-raw", type=float, default=0.45,
                        help="If local_pair_raw_score >= this, motorcycle_context will not hard-veto.")
    ap.add_argument("--twowheel-exempt-contact-frames", type=int, default=3,
                        help="If max_pair_contact_frames >= this, motorcycle_context will not hard-veto.")
    ap.add_argument("--twowheel-exempt-close-frames", type=int, default=8,
                        help="If max_pair_close_frames >= this, motorcycle_context will not hard-veto.")
    ap.add_argument("--enable-area-relaxed-retry", action="store_true",
                        help="When no_pair_after_area_filter and pair evidence exists, retry with smaller min_area_ratio.")
    ap.add_argument("--area-relaxed-ratio-mul", type=float, default=0.5,
                        help="Retry min_area_ratio = min_area_ratio * this multiplier.")
    ap.add_argument("--area-relaxed-min-area-ratio", type=float, default=0.0004,
                        help="Lower bound for relaxed retry min_area_ratio.")
    ap.add_argument("--area-relaxed-min-contact-frames", type=int, default=3,
                        help="Enable relaxed retry when max_pair_contact_frames >= this.")
    ap.add_argument("--area-relaxed-min-close-frames", type=int, default=8,
                        help="Enable relaxed retry when max_pair_close_frames >= this.")
    ap.add_argument("--person-bonus", type=float, default=0.0,
                        help="Add this bonus to pair score when person_peak >= person-bonus-min-peak. 0=disabled.")
    ap.add_argument("--person-bonus-min-peak", type=int, default=3,
                        help="Minimum person_peak to trigger person-bonus.")
    ap.add_argument("--congestion-strong-rel-compact", type=float, default=99.0,
                        help="For base=拥堵: if strong_channel AND best_pair.relative_compactness >= this, also apply flip. Default 99.0 (disabled).")
    ap.add_argument("--congestion-strong-min-sc", type=int, default=0,
                        help="congestion_strong channel: static_count must be >= this. 0=disabled.")
    ap.add_argument("--congestion-strong-max-compact", type=float, default=1.0,
                        help="congestion_strong channel: compactness must be <= this (upper bound). 1.0=disabled.")
    ap.add_argument("--cone-peak", type=int, default=8)
    ap.add_argument("--construction-vehicle-peak", type=int, default=4)
    ap.add_argument("--cone-pair-peak", type=int, default=5)
    ap.add_argument("--barrier-pair-peak", type=int, default=3)
    ap.add_argument("--block-debris-context", action="store_true")
    ap.add_argument("--debris-peak", type=int, default=1)
    ap.add_argument("--road-obstacle-peak", type=int, default=3)
    ap.add_argument("--vscm-debug-score-thr", type=float, default=0.70)
    ap.add_argument("--vscm-relation-norm", type=float, default=6.0)
    ap.add_argument("--vscm-context-norm", type=float, default=6.0)
    ap.add_argument("--vscm-queue-penalty", type=float, default=0.15)
    ap.add_argument("--vscm-min-stagnation-frames", type=int, default=4)
    ap.add_argument("--vscm-min-contact-frames", type=int, default=2)
    # ── Semantic text gate (low-cost; uses existing E8 pred text) ──────────────
    ap.add_argument("--text-semantic-mode", default="off",
                        choices=["off", "veto", "confirm", "balanced"],
                        help="Semantic text gate mode. "
                             "'veto': suppress LCRM when negative keywords present and no positive evidence. "
                             "'confirm': require at least one positive keyword or OVD context hit. "
                             "'balanced': veto OR require confirmation. "
                             "'off' (default): disabled, no change to current behaviour.")
    ap.add_argument("--text-positive-min", type=int, default=1,
                        help="Min combined positive keyword hits (accident + scene) to count as text-confirmed.")
    ap.add_argument("--text-neg-veto-min", type=int, default=1,
                        help="Min negative keyword hits to trigger a text veto.")
    ap.add_argument("--semantic-person-min-peak", type=int, default=2,
                        help="person_peak >= this counts as OVD context-confirmed.")
    ap.add_argument("--semantic-contact-min-frames", type=int, default=3,
                        help="max_pair_contact_frames >= this counts as OVD context-confirmed.")
    ap.add_argument("--semantic-close-min-frames", type=int, default=6,
                        help="max_pair_close_frames >= this counts as OVD context-confirmed.")
    ap.add_argument("--semantic-confirm-score", type=float, default=0.80,
                        help="best_pair.score >= this (together with compactness and persistence thresholds) counts as geometry-confirmed.")
    ap.add_argument("--semantic-confirm-compactness", type=float, default=0.55,
                        help="best_pair.compactness >= this for geometry-confirmed path.")
    ap.add_argument("--semantic-confirm-persistence", type=int, default=6,
                        help="best_pair.persistence_frames >= this for geometry-confirmed path.")
    args = ap.parse_args()

    labels = parse_csv(args.labels)
    if labels != DEFAULT_LABELS:
        raise ValueError(
            "--labels must match eval_traffic_fixed.py LABELS exactly. "
            f"got={labels}, expected={DEFAULT_LABELS}"
        )
    base_rows = read_jsonl(args.base_pred)
    ovd_map = by_video(read_jsonl(args.ovd))

    out_rows = []
    stats = Counter()
    changes_by_gt = Counter()
    signal_by_gt = Counter()
    reason_by_gt = Counter()
    detail_by_gt = Counter()
    dgroup_debug = []
    # Resolve construction bypass score for congestion base (None → inherit general threshold)
    _cb_score_cong = (
        args.construction_bypass_score_congestion
        if args.construction_bypass_score_congestion is not None
        else args.construction_bypass_score
    )

    for row in base_rows:
        video = row.get("video", "")
        gt = label_from_video(video)
        base_text = pick_pred_text(row, args.base_pred_field)
        base_label = detect_label(base_text)
        ovd = get_ovd_payload(ovd_map.get(video, {}))
        lcrm = ovd.get("lcrm") if isinstance(ovd, dict) else None
        vscm_debug = ovd.get("vscm_debug") if isinstance(ovd, dict) else None

        stats["total"] += 1
        if ovd:
            stats["ovd_matched"] += 1
        if isinstance(lcrm, dict):
            stats["lcrm_present"] += 1
            if lcrm.get("ok"):
                stats["lcrm_ok"] += 1
        if isinstance(vscm_debug, dict):
            stats["vscm_debug_present"] += 1
        if base_label:
            stats["base_label_found"] += 1
        else:
            stats["base_label_missing"] += 1

        if args.source == "lcrm":
            signal, debug = lcrm_lite_signal(lcrm, args)
        elif args.source == "vscm_debug":
            signal, debug = vscm_debug_lite_signal(vscm_debug, args)
        elif isinstance(lcrm, dict):
            signal, debug = lcrm_lite_signal(lcrm, args)
            debug["source"] = "lcrm"
        else:
            signal, debug = vscm_debug_lite_signal(vscm_debug, args)
            debug["source"] = "vscm_debug"
        reason_by_gt[(gt, debug.get("reason"))] += 1
        # Optional semantic text gate: uses existing E8 pred text + OVD person context.
        # Only active when --text-semantic-mode != 'off'.
        signal, debug = apply_semantic_text_gate(signal, debug, lcrm, args, base_text)
        if base_label == MULTICAR_LABEL:
            _bp = debug.get("best_pair") or {}
            dgroup_debug.append({
                "video": video,
                "gt": gt,
                "signal": signal,
                "cac": debug.get("congestion_accident_channel", False),
                "sc": debug.get("static_count", 0),
                "compact": _bp.get("compactness"),
                "persist": _bp.get("persistence_frames"),
                "rel_compact": _bp.get("relative_compactness"),
                "reason": debug.get("reason"),
            })
        fused_label = base_label
        can_try = base_label in args.allowed_base_labels
        if (
            not can_try
            and signal
            and debug.get("construction_bypass", False)
            and base_label in args.construction_bypass_extra_base_labels
        ):
            can_try = True

        if can_try and signal:
            # Per-label routing: congestion base requires the stricter
            # congestion_accident_channel instead of the standard channels.
            if base_label == CONGESTION_LABEL:
                _bpair = debug.get("best_pair") or {}
                _brel = _bpair.get("relative_compactness")
                congestion_strong = (
                    args.congestion_strong_rel_compact < 90.0
                    and debug.get("channel") == "strong"
                    and _brel is not None
                    and _brel >= args.congestion_strong_rel_compact
                    and (args.congestion_strong_min_sc <= 0 or debug.get("static_count", 0) >= args.congestion_strong_min_sc)
                    and (_bpair.get("compactness", 1.0) or 1.0) <= args.congestion_strong_max_compact
                )
                _cong_bypass_ok = (
                    debug.get("construction_bypass", False)
                    and (debug.get("best_pair") or {}).get("score", 0) >= _cb_score_cong
                )
                apply = (
                    debug.get("congestion_accident_channel", False)
                    or _cong_bypass_ok
                    or congestion_strong
                )
            else:
                apply = True
            if apply:
                fused_label = MULTICAR_LABEL
                signal_by_gt[gt] += 1

        fused_text = fused_label if fused_label and fused_label != base_label else base_text
        # D-group normalization: when E8 already predicted 多车事故 (base=MULTICAR) and the
        # congestion_accident_channel fires (compact≥0.65, persist≥10, rel_compact OK, sc≤10),
        # override fused_text to MULTICAR_LABEL so eval cannot "auto-correct" it to 拥堵.
        # Safe: all GT=拥堵 D-group records have cac=False (verified on smoke-240 set).
        if (
            base_label == MULTICAR_LABEL
            and signal
            and debug.get("congestion_accident_channel", False)
        ):
            fused_text = MULTICAR_LABEL
        rec = dict(row)
        rec[args.fused_field] = fused_text
        rec["lcrm_lite_base_label"] = base_label
        rec["lcrm_lite_fused_label"] = fused_label
        rec["lcrm_lite_signal"] = bool(base_label in args.allowed_base_labels and signal)
        rec["lcrm_lite_debug"] = debug
        out_rows.append(rec)

        detail_by_gt[(gt, base_label, fused_label, debug.get("reason"))] += 1
        if fused_label and fused_label != base_label:
            stats["changed"] += 1
            changes_by_gt[gt] += 1

    write_jsonl(args.output, out_rows)
    print("output:", args.output)
    print("stats:", dict(stats))
    print("signal_by_gt:", dict(signal_by_gt))
    print("changes_by_gt:", dict(changes_by_gt))
    print("reason_by_gt:", {str(k): v for k, v in reason_by_gt.items()})
    print("detail_by_gt:", {str(k): v for k, v in detail_by_gt.items()})
    if dgroup_debug:
        print("dgroup_debug:", json.dumps(dgroup_debug, ensure_ascii=False))


if __name__ == "__main__":
    main()
