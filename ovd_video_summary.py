import json
from collections import Counter
from pathlib import Path
import cv2
import torch
try:
    from transformers import GroundingDinoProcessor as _Proc, GroundingDinoForObjectDetection as _Model
    _USE_GROUNDING_DINO_CLASS = True
except ImportError:
    from transformers import AutoProcessor as _Proc, AutoModelForZeroShotObjectDetection as _Model
    _USE_GROUNDING_DINO_CLASS = False

TEXT_QUERIES = [
    "car", "truck", "bus", "motorcycle", "person",
    "traffic cone", "construction vehicle", "road barrier",
    "debris", "road obstacle", "scattered object",
]

VEHICLE_QUERIES = {"car", "truck", "bus"}
VSCM_CONTEXT_QUERIES = {"person", "traffic cone", "construction vehicle", "road barrier"}
SARP_TARGET_QUERIES = {
    "debris", "road debris", "road obstacle", "scattered object",
    "road spill", "trash",
}
SARP_NEGATIVE_QUERIES = {
    "car", "truck", "bus", "motorcycle", "person",
    "traffic cone", "construction vehicle", "road barrier",
}
SARP_ENHANCED_QUERIES = sorted(SARP_NEGATIVE_QUERIES | SARP_TARGET_QUERIES)

def sample_indices(n, k=12):
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    step = n / k
    return [int(i * step) for i in range(k)]

def _box_iou(a, b):
    """IoU of two boxes [x1,y1,x2,y2]."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _box_center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _dedupe_vehicle_boxes(boxes, iou_thresh=0.50, center_dist_frac=0.18):
    """Merge duplicate vehicle boxes from open-vocabulary detections.

    GroundingDINO may emit several boxes for the same vehicle across car/truck/
    bus prompts. Without this pass, downstream vehicle evidence can be inflated
    by duplicate class prompts.
    """
    if not boxes:
        return []

    kept = []
    for box in sorted(boxes, key=_box_area, reverse=True):
        cx, cy = _box_center(box)
        w = max(1.0, box[2] - box[0])
        h = max(1.0, box[3] - box[1])
        duplicate = False
        for old in kept:
            ocx, ocy = _box_center(old)
            ow = max(1.0, old[2] - old[0])
            oh = max(1.0, old[3] - old[1])
            avg_w = (w + ow) / 2
            avg_h = (h + oh) / 2
            center_close = (
                abs(cx - ocx) < center_dist_frac * avg_w
                and abs(cy - ocy) < center_dist_frac * avg_h
            )
            size_similar = (
                min(w, ow) / max(w, ow) > 0.55
                and min(h, oh) / max(h, oh) > 0.55
            )
            if _box_iou(box, old) >= iou_thresh or (center_close and size_similar):
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return kept


def _box_edge_proximity(a, b):
    """
    闂佸搫顦弲婊堝蓟閵娿儍?(edge_gap, x_overlap_ratio, y_overlap_ratio)闂?
    edge_gap: 濠电偞鍨堕幐鍫曞磹閺嶎厹鈧胶鈧綆鍋呯紞鍥煕閿旇骞橀柛鐘筹耿閺岋繝宕煎顑垮闂佽绻愮换鎰崲濡ゅ啰鐜绘繛鎴炴皑閻霉閸忚偐鎳呯紒鈧崟顖涒拺闁圭粯甯炲瓭閻熸粎澧楀Λ鍐嚕?0闂備焦瀵х粙鎴λ囬銏犵劦?
    x/y_overlap_ratio: 濠电偞鍨堕幐鍫曞磹閺嶎厹鈧胶鈧綆鍠栭幑鍫曟煏婵犲海鍘涢柛銈咁樀瀵爼鎮欓悽鐢电槇缂備焦顨呴崐鍧楀蓟鐏炵瓔鍚嬮柛娑卞幘缁夐箖姊婚崒妤€浜鹃梺鍓茬厛閸犳牠顢?/ 闂佸搫顦悧鍡涘疮椤愶附鍎嶆い鎺嗗亾妞も晛銈稿畷鍗炍旀担钘夋儓闂佽崵濮村ú銏ゅ磿閹绢喗鐓€閺夊牄鍔庨埢鏂库攽閻樻彃顏柣锝変憾濮婂鍩€椤掑嫬鍨傛い鏃傗拡閸炴椽姊?
    """
    x_gap = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    y_gap = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    edge_gap = (x_gap ** 2 + y_gap ** 2) ** 0.5

    x_overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    y_overlap = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    wa = a[2] - a[0]; wb = b[2] - b[0]
    ha = a[3] - a[1]; hb = b[3] - b[1]
    x_overlap_ratio = x_overlap / min(wa, wb) if min(wa, wb) > 0 else 0.0
    y_overlap_ratio = y_overlap / min(ha, hb) if min(ha, hb) > 0 else 0.0

    return edge_gap, x_overlap_ratio, y_overlap_ratio


def _box_intersects(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _point_in_box(point, box):
    x, y = point
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _expand_box(b, dx, dy, img_width, img_height):
    return [
        max(0.0, b[0] - dx),
        max(0.0, b[1] - dy),
        min(float(img_width), b[2] + dx),
        min(float(img_height), b[3] + dy),
    ]


def _union_boxes(boxes):
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def _vscm_debug(reason, **kwargs):
    data = {
        "debug_only": True,
        "none_reason": reason,
        "triggered": False,
    }
    data.update(kwargs)
    return data


def _build_vehicle_tracks(per_frame_vehicle_boxes, img_width, min_track_len=3, match_dist_frac=0.08):
    """Short-track vehicle matching for sampled frames."""
    tracks = []
    if not per_frame_vehicle_boxes or img_width <= 0:
        return tracks

    for frame_idx, frame_boxes in enumerate(per_frame_vehicle_boxes):
        unmatched = list(frame_boxes or [])
        for track in tracks:
            if not unmatched:
                break
            last_c = _box_center(track[-1][1])
            dists = [
                (((_box_center(b)[0] - last_c[0]) ** 2 + (_box_center(b)[1] - last_c[1]) ** 2) ** 0.5)
                for b in unmatched
            ]
            best = int(min(range(len(dists)), key=lambda i: dists[i]))
            if dists[best] / img_width < match_dist_frac:
                track.append((frame_idx, unmatched.pop(best)))
        for box in unmatched:
            tracks.append([(frame_idx, box)])

    return [track for track in tracks if len(track) >= min_track_len]


def _track_motion_stats(track, img_width):
    centers = [_box_center(box) for _, box in track]
    if len(centers) <= 1 or img_width <= 0:
        return {"max_step": 0.0, "total_disp": 0.0, "max_step_norm": 0.0, "total_disp_norm": 0.0}
    steps = [
        ((centers[i][0] - centers[i - 1][0]) ** 2 + (centers[i][1] - centers[i - 1][1]) ** 2) ** 0.5
        for i in range(1, len(centers))
    ]
    total_disp = ((centers[-1][0] - centers[0][0]) ** 2 + (centers[-1][1] - centers[0][1]) ** 2) ** 0.5
    max_step = max(steps) if steps else 0.0
    return {
        "max_step": max_step,
        "total_disp": total_disp,
        "max_step_norm": max_step / img_width,
        "total_disp_norm": total_disp / img_width,
    }


def _clip01(x):
    return max(0.0, min(1.0, float(x)))


def _compute_lcrm_v1(per_frame_vehicle_boxes, img_width, img_height, labels_peak=None, labels_all=None):
    """
    LCRM: Local Congestion/Relation Motion diagnostic signals.

    This module intentionally returns scores and raw structure features instead
    of a hard class label. It separates:
      - close local static pairs: multi-car accident-like evidence
      - isolated static vehicles: abnormal-parking-like evidence
      - distributed slow/static regions: congestion-like evidence
    """
    labels_peak = labels_peak or {}
    labels_all = labels_all or {}
    if not per_frame_vehicle_boxes or img_width <= 0 or img_height <= 0:
        return {
            "ok": False,
            "reason": "no_frame_or_invalid_size",
            "frame_count": len(per_frame_vehicle_boxes or []),
        }

    STATIC_STEP = 0.04
    SLOW_STEP = 0.10
    tracks = _build_vehicle_tracks(per_frame_vehicle_boxes, img_width, min_track_len=3, match_dist_frac=0.08)
    if not tracks:
        return {
            "ok": False,
            "reason": "no_tracks",
            "frame_count": len(per_frame_vehicle_boxes or []),
            "vehicle_peak": max((len(x or []) for x in per_frame_vehicle_boxes), default=0),
        }

    static_tracks = []
    slow_tracks = []
    moving_tracks = []
    track_infos = []
    for tid, track in enumerate(tracks):
        stats = _track_motion_stats(track, img_width)
        if stats["max_step_norm"] < STATIC_STEP:
            bucket = "static"
            static_tracks.append(track)
        elif stats["max_step_norm"] < SLOW_STEP:
            bucket = "slow"
            slow_tracks.append(track)
        else:
            bucket = "moving"
            moving_tracks.append(track)
        last_box = track[-1][1]
        cx, cy = _box_center(last_box)
        track_infos.append({
            "track_id": tid,
            "bucket": bucket,
            "frames": len(track),
            "max_step_norm": round(stats["max_step_norm"], 4),
            "total_disp_norm": round(stats["total_disp_norm"], 4),
            "last_center": [round(cx / img_width, 3), round(cy / img_height, 3)],
            "last_box": [round(x, 1) for x in last_box],
        })

    static_count = len(static_tracks)
    slow_count = len(slow_tracks)
    moving_count = len(moving_tracks)
    track_count = len(tracks)
    slow_like_count = static_count + slow_count
    global_slow_ratio = slow_like_count / max(1, track_count)
    vehicle_peak = max((len(x or []) for x in per_frame_vehicle_boxes), default=0)
    motorcycle_peak = int(labels_peak.get("motorcycle", 0) or 0)
    traffic_cone_peak = int(labels_peak.get("traffic cone", 0) or 0)
    construction_vehicle_peak = int(labels_peak.get("construction vehicle", 0) or 0)
    road_barrier_peak = int(labels_peak.get("road barrier", 0) or 0)
    debris_peak = int(labels_peak.get("debris", 0) or 0)
    road_obstacle_peak = int(labels_peak.get("road obstacle", 0) or 0)

    min_static_pair_edge_gap_norm = None
    min_static_pair_edge_gap_width = None
    max_pair_close_frames = 0
    max_pair_contact_frames = 0
    best_pair = None

    for i in range(len(static_tracks)):
        boxes_i = {frame_idx: box for frame_idx, box in static_tracks[i]}
        for j in range(i + 1, len(static_tracks)):
            boxes_j = {frame_idx: box for frame_idx, box in static_tracks[j]}
            common_frames = sorted(set(boxes_i) & set(boxes_j))
            close_frames = 0
            contact_frames = 0
            pair_min_gap_norm = None
            pair_min_gap_width = None
            for frame_idx in common_frames:
                a = boxes_i[frame_idx]
                b = boxes_j[frame_idx]
                edge_gap, x_overlap, y_overlap = _box_edge_proximity(a, b)
                avg_width = max(1.0, ((a[2] - a[0]) + (b[2] - b[0])) / 2)
                gap_norm = edge_gap / img_width
                gap_width = edge_gap / avg_width
                pair_min_gap_norm = gap_norm if pair_min_gap_norm is None else min(pair_min_gap_norm, gap_norm)
                pair_min_gap_width = gap_width if pair_min_gap_width is None else min(pair_min_gap_width, gap_width)
                contact = _box_iou(a, b) > 0.05 or (
                    gap_width < 0.12 and (x_overlap > 0.45 or y_overlap > 0.45)
                )
                close = gap_norm < 0.045 or (
                    gap_width < 0.65 and (x_overlap > 0.35 or y_overlap > 0.35)
                )
                if contact:
                    contact_frames += 1
                if close:
                    close_frames += 1

            if pair_min_gap_norm is not None:
                if min_static_pair_edge_gap_norm is None or pair_min_gap_norm < min_static_pair_edge_gap_norm:
                    min_static_pair_edge_gap_norm = pair_min_gap_norm
                    min_static_pair_edge_gap_width = pair_min_gap_width
                    best_pair = [i, j]
            max_pair_close_frames = max(max_pair_close_frames, close_frames)
            max_pair_contact_frames = max(max_pair_contact_frames, contact_frames)

    # Spatial spread of slow/static tracks. Congestion is more likely when
    # slow-like vehicles occupy multiple coarse regions instead of one cluster.
    slow_regions = set()
    slow_region_hits = Counter()
    for track in static_tracks + slow_tracks:
        for frame_idx, box in track:
            cx, cy = _box_center(box)
            rx = min(2, max(0, int(cx / img_width * 3)))
            ry = min(2, max(0, int(cy / img_height * 3)))
            key = f"{rx},{ry}"
            slow_regions.add(key)
            slow_region_hits[key] += 1
    slow_region_count = len(slow_regions)
    dominant_region_ratio = (
        max(slow_region_hits.values()) / max(1, sum(slow_region_hits.values()))
        if slow_region_hits else 0.0
    )

    construction_context = (
        traffic_cone_peak >= 3
        or construction_vehicle_peak >= 2
        or (road_barrier_peak >= 3 and traffic_cone_peak >= 1)
    )
    debris_context = debris_peak >= 1 or road_obstacle_peak >= 2
    motorcycle_context = motorcycle_peak >= 1

    # Pair score favors 2-4 static vehicles, persistent close relation, and a
    # background that is not globally slow everywhere. Semantic suppressors are
    # soft: they reduce competing category false positives without deleting the
    # diagnostic feature entirely.
    close_term = min(1.0, max_pair_close_frames / 5.0)
    # count_term also gives partial credit for 5-6 static vehicles (wider range)
    count_term = 1.0 if 2 <= static_count <= 4 else (0.6 if static_count == 5 or static_count == 6 else (0.3 if static_count == 1 else 0.0))
    flow_term = _clip01(moving_count / 4.0)
    non_global_term = _clip01((0.90 - global_slow_ratio) / 0.45)
    # v3: even without contact frames, count+flow+non_global can reach ~0.55
    # so lower the threshold to 0.32 to allow contact-free accident patterns
    local_pair_raw = _clip01(0.45 * close_term + 0.25 * count_term + 0.20 * flow_term + 0.10 * non_global_term)
    local_pair_suppression = 1.0
    if motorcycle_context:
        local_pair_suppression *= 0.55
    if construction_context:
        local_pair_suppression *= 0.75
    if debris_context:
        local_pair_suppression *= 0.85
    local_pair_score = round(_clip01(local_pair_raw * local_pair_suppression), 3)

    # Isolated stop score favors one static vehicle or far-apart static vehicles
    # while surrounding tracks still move.
    if min_static_pair_edge_gap_norm is None:
        isolation_term = 1.0
    else:
        isolation_term = _clip01((min_static_pair_edge_gap_norm - 0.04) / 0.16)
    # v3: relax to static_count <= 2 (parking often shows 1-2 stopped vehicles)
    isolated_count_term = 1.0 if static_count == 1 else (0.65 if static_count == 2 else 0.0)
    isolated_raw = _clip01(
        0.40 * isolated_count_term + 0.25 * isolation_term + 0.25 * flow_term + 0.10 * non_global_term
    )
    isolated_suppression = 1.0
    if construction_context:
        isolated_suppression *= 0.45
    if debris_context:
        isolated_suppression *= 0.70
    if motorcycle_context:
        isolated_suppression *= 0.80
    isolated_stop_score = round(_clip01(isolated_raw * isolated_suppression), 3)

    # Distributed congestion must be dense and spatially spread. v1 treated
    # almost every stable highway scene as congestion; v2 gates the score on
    # vehicle count and region spread before scoring.
    region_term = _clip01((slow_region_count - 2) / 4.0)
    density_term = _clip01((vehicle_peak - 5) / 10.0)
    slow_term = _clip01((global_slow_ratio - 0.60) / 0.35)
    spread_term = _clip01((0.70 - dominant_region_ratio) / 0.45)
    track_term = _clip01((track_count - 6) / 10.0)
    congestion_raw = _clip01(
        0.30 * slow_term + 0.25 * density_term + 0.20 * region_term + 0.15 * spread_term + 0.10 * track_term
    )
    congestion_gate = (
        vehicle_peak >= 7
        and track_count >= 7
        and slow_region_count >= 3
        and dominant_region_ratio <= 0.75
        and global_slow_ratio >= 0.60
    )
    distributed_congestion_score = round(congestion_raw if congestion_gate else min(congestion_raw, 0.49), 3)

    primary = "none"
    scores = {
        "local_pair": local_pair_score,
        "isolated_stop": isolated_stop_score,
        "distributed_congestion": distributed_congestion_score,
    }
    # v3: distributed_congestion is too indiscriminate across all traffic classes
    # (fires at 70-95% for all categories). Remove it from primary_signal decision.
    # Keep it in scores dict for statistics only.
    # local_pair threshold lowered to 0.32: allows count+flow to trigger even
    # without bbox contact, catching accidents where cars stop apart.
    if local_pair_score >= 0.32 and local_pair_score >= isolated_stop_score:
        primary = "local_pair"
    elif isolated_stop_score >= 0.32 and isolated_stop_score > local_pair_score:
        primary = "isolated_stop"

    return {
        "ok": True,
        "version": "v2",
        "primary_signal": primary,
        "local_pair_score": local_pair_score,
        "isolated_stop_score": isolated_stop_score,
        "distributed_congestion_score": distributed_congestion_score,
        "local_pair_raw_score": round(local_pair_raw, 3),
        "isolated_stop_raw_score": round(isolated_raw, 3),
        "distributed_congestion_raw_score": round(congestion_raw, 3),
        "track_count": track_count,
        "static_count": static_count,
        "slow_count": slow_count,
        "moving_count": moving_count,
        "global_slow_ratio": round(global_slow_ratio, 3),
        "vehicle_peak": vehicle_peak,
        "semantic_context": {
            "motorcycle_peak": motorcycle_peak,
            "traffic_cone_peak": traffic_cone_peak,
            "construction_vehicle_peak": construction_vehicle_peak,
            "road_barrier_peak": road_barrier_peak,
            "debris_peak": debris_peak,
            "road_obstacle_peak": road_obstacle_peak,
            "motorcycle_context": motorcycle_context,
            "construction_context": construction_context,
            "debris_context": debris_context,
        },
        "slow_region_count": slow_region_count,
        "dominant_slow_region_ratio": round(dominant_region_ratio, 3),
        "min_static_pair_edge_gap_norm": (
            None if min_static_pair_edge_gap_norm is None else round(min_static_pair_edge_gap_norm, 4)
        ),
        "min_static_pair_edge_gap_width": (
            None if min_static_pair_edge_gap_width is None else round(min_static_pair_edge_gap_width, 3)
        ),
        "max_pair_close_frames": max_pair_close_frames,
        "max_pair_contact_frames": max_pair_contact_frames,
        "best_static_pair": best_pair,
        "tracks": sorted(track_infos, key=lambda x: (x["bucket"] != "static", -x["frames"]))[:12],
    }


def _compute_vscm_v3(per_frame_vehicle_boxes, per_frame_context_boxes, img_width, img_height):
    """
    VSCM for visible multi-car accident evidence.

    v3 changes the decision rule:
      - bbox contact is only auxiliary evidence, never a standalone trigger
      - the static vehicle cluster must have nearby accident-scene evidence
        (person/cone/barrier/construction vehicle) or a strong persistent pair
      - dense queue context suppresses weak contact-only cases
    """
    if not per_frame_vehicle_boxes or img_width <= 0:
        return _vscm_debug(
            "no_frame_or_invalid_size",
            frame_count=len(per_frame_vehicle_boxes or []),
            img_width=img_width,
            img_height=img_height,
        )

    # v3.9 candidate-generation pass:
    # - 0.03/0.06 was very high precision but dropped many true accident
    #   candidates before the final VSCM rules could inspect them.
    # - Slightly relax both thresholds, while keeping the final trigger rules
    #   conservative.
    STAGNATION_THRESH = 0.04
    MATCH_DIST_FRAC = 0.08
    MIN_TRACK_LEN = 3
    MIN_CONTACT_FRAMES = 3
    MIN_CLOSE_FRAMES = 5
    MIN_PAIR_STAGNATION_FRAMES = 7
    MIN_CONTEXT_FRAMES = 2

    tracks = []  # list[list[tuple[int, box]]]
    for frame_idx, frame_boxes in enumerate(per_frame_vehicle_boxes):
        unmatched = list(frame_boxes)
        for track in tracks:
            if not unmatched:
                break
            last_c = _box_center(track[-1][1])
            dists = [(((_box_center(b)[0] - last_c[0]) ** 2 +
                       (_box_center(b)[1] - last_c[1]) ** 2) ** 0.5)
                     for b in unmatched]
            best = int(min(range(len(dists)), key=lambda i: dists[i]))
            if dists[best] / img_width < MATCH_DIST_FRAC:
                track.append((frame_idx, unmatched.pop(best)))
        for b in unmatched:
            tracks.append([(frame_idx, b)])

    static_tracks = []
    moving_count = 0
    for track in tracks:
        if len(track) < MIN_TRACK_LEN:
            continue
        centers = [_box_center(b) for _, b in track]
        max_disp = max(
            ((centers[i][0] - centers[i - 1][0]) ** 2 +
             (centers[i][1] - centers[i - 1][1]) ** 2) ** 0.5
            for i in range(1, len(centers))
        )
        if max_disp / img_width < STAGNATION_THRESH:
            static_tracks.append(track)
        else:
            moving_count += 1

    static_count = len(static_tracks)
    total_count = static_count + moving_count

    if total_count < 5:
        return _vscm_debug(
            "total_count_lt_5",
            static_count=static_count,
            moving_count=moving_count,
            total_count=total_count,
        )
    if not (2 <= static_count <= 6):
        return _vscm_debug(
            "static_count_out_2_6",
            static_count=static_count,
            moving_count=moving_count,
            total_count=total_count,
        )
    static_ratio = static_count / total_count
    if static_ratio > 0.28:
        return _vscm_debug(
            "static_ratio_gt_0.28",
            static_count=static_count,
            moving_count=moving_count,
            total_count=total_count,
            static_ratio=round(static_ratio, 3),
        )
    if moving_count < 5:
        return _vscm_debug(
            "moving_count_lt_5",
            static_count=static_count,
            moving_count=moving_count,
            total_count=total_count,
            static_ratio=round(static_ratio, 3),
        )

    static_last_boxes = [track[-1][1] for track in static_tracks]
    avg_box_width = sum((b[2] - b[0]) for b in static_last_boxes) / len(static_last_boxes)

    # Many abnormal-parking false positives happen at the image edge/shoulder.
    # Keep this weak: only reject clusters whose center is extremely close to
    # the left/right image border.
    cluster_center_x = sum(_box_center(b)[0] for b in static_last_boxes) / len(static_last_boxes)
    cluster_center_x_ratio = cluster_center_x / img_width
    if cluster_center_x_ratio < 0.12 or cluster_center_x_ratio > 0.88:
        return _vscm_debug(
            "cluster_center_out_0.12_0.88",
            static_count=static_count,
            moving_count=moving_count,
            total_count=total_count,
            static_ratio=round(static_ratio, 3),
            cluster_center_x_ratio=round(cluster_center_x_ratio, 3),
        )

    relation_evidence = False
    proximity_reason = None
    max_contact_frames = 0
    max_close_frames = 0
    max_pair_stagnation_frames = 0
    static_pair_stagnation = False

    for i in range(len(static_tracks)):
        boxes_i = {frame_idx: box for frame_idx, box in static_tracks[i]}
        for j in range(i + 1, len(static_tracks)):
            boxes_j = {frame_idx: box for frame_idx, box in static_tracks[j]}
            common_frames = sorted(set(boxes_i) & set(boxes_j))
            contact_frames = 0
            close_frames = 0
            pair_stagnation_frames = 0
            reason = None
            for frame_idx in common_frames:
                a = boxes_i[frame_idx]
                b = boxes_j[frame_idx]
                pair_avg_width = max(1.0, ((a[2] - a[0]) + (b[2] - b[0])) / 2)
                iou = _box_iou(a, b)
                edge_gap, x_overlap, y_overlap = _box_edge_proximity(a, b)

                contact = (
                    iou > 0.07
                    or (
                        edge_gap < 0.10 * pair_avg_width
                        and (x_overlap > 0.55 or y_overlap > 0.55)
                    )
                )
                # Persistent close spacing: captures two static accident vehicles
                # that do not overlap, while requiring temporal persistence to
                # suppress one-frame dense-traffic false positives.
                close_static_pair = (
                    static_count <= 3
                    and static_ratio <= 0.18
                    and edge_gap < 0.55 * pair_avg_width
                    and (x_overlap > 0.50 or y_overlap > 0.50)
                )
                # Accident-after-stagnation pattern: two stopped vehicles can
                # remain separated after a collision/incident, so requiring
                # bbox contact misses many true multi-car cases. Keep this as
                # a conservative pair-level cue: the same two static tracks must
                # persist for several sampled frames and stay in a plausible
                # same/adjacent-lane neighborhood.
                ca = _box_center(a)
                cb = _box_center(b)
                center_dx = abs(ca[0] - cb[0])
                center_dy = abs(ca[1] - cb[1])
                same_or_adjacent_lane = (
                    x_overlap > 0.20
                    or center_dx < 1.25 * pair_avg_width
                )
                separated_but_related = (
                    edge_gap < 2.50 * pair_avg_width
                    and center_dy < 3.00 * pair_avg_width
                    and same_or_adjacent_lane
                )

                if contact:
                    contact_frames += 1
                    reason = 'contact'
                if close_static_pair:
                    close_frames += 1
                    if reason is None:
                        reason = 'persistent_close_pair'
                if separated_but_related:
                    pair_stagnation_frames += 1

            max_contact_frames = max(max_contact_frames, contact_frames)
            max_close_frames = max(max_close_frames, close_frames)
            max_pair_stagnation_frames = max(max_pair_stagnation_frames, pair_stagnation_frames)
            if pair_stagnation_frames >= MIN_PAIR_STAGNATION_FRAMES:
                static_pair_stagnation = True
            if contact_frames >= MIN_CONTACT_FRAMES or close_frames >= MIN_CLOSE_FRAMES:
                relation_evidence = True
                proximity_reason = 'contact' if contact_frames >= MIN_CONTACT_FRAMES else 'persistent_close_pair'
                break
        if relation_evidence:
            break

    # Context evidence around the static vehicle cluster. This is the main
    # guard against treating ordinary queue/parking bbox contact as accident.
    static_by_frame = {}
    for track in static_tracks:
        for frame_idx, box in track:
            static_by_frame.setdefault(frame_idx, []).append(box)

    context_hits = {k: 0 for k in VSCM_CONTEXT_QUERIES}
    context_frames = 0
    queue_like_frames = 0
    checked_frames = 0

    for frame_idx, static_boxes in static_by_frame.items():
        if not static_boxes:
            continue
        checked_frames += 1
        cluster = _union_boxes(static_boxes)
        widths = [max(1.0, b[2] - b[0]) for b in static_boxes]
        heights = [max(1.0, b[3] - b[1]) for b in static_boxes]
        avg_w = sum(widths) / len(widths)
        avg_h = sum(heights) / len(heights)

        context_area = _expand_box(cluster, 1.0 * avg_w, 0.8 * avg_h, img_width, img_height)
        queue_area = _expand_box(cluster, 1.8 * avg_w, 0.9 * avg_h, img_width, img_height)

        frame_context_found = False
        context_boxes = per_frame_context_boxes[frame_idx] if frame_idx < len(per_frame_context_boxes) else {}
        for label in VSCM_CONTEXT_QUERIES:
            for box in context_boxes.get(label, []):
                if _box_intersects(context_area, box):
                    context_hits[label] += 1
                    frame_context_found = True

        if frame_context_found:
            context_frames += 1

        # Dense queue suppression: many vehicles packed around the same local
        # cluster is more likely congestion/queue than a localized accident.
        frame_boxes = per_frame_vehicle_boxes[frame_idx] if frame_idx < len(per_frame_vehicle_boxes) else []
        local_vehicle_count = 0
        for box in frame_boxes:
            cx, cy = _box_center(box)
            if queue_area[0] <= cx <= queue_area[2] and queue_area[1] <= cy <= queue_area[3]:
                local_vehicle_count += 1
        if local_vehicle_count >= static_count + 5:
            queue_like_frames += 1

    has_context = context_frames >= MIN_CONTEXT_FRAMES
    # Road barriers are common in highway scenes and GroundingDINO often fires
    # on normal roadside structures, so they must not be standalone evidence.
    meaningful_scene_context = (
        context_hits.get("person", 0) >= 2
        or (
            context_hits.get("construction vehicle", 0) >= 3
            and context_hits.get("person", 0) >= 1
        )
        or (
            context_hits.get("traffic cone", 0) >= 3
            and (context_hits.get("person", 0) >= 1 or context_hits.get("construction vehicle", 0) >= 2)
        )
    )
    queue_like = checked_frames > 0 and queue_like_frames / checked_frames >= 0.45

    # Trigger only when the static cluster is supported by scene evidence.
    # The old persistent-close-pair fallback caused two-wheel false positives,
    # so v3.4 keeps close/contact as relation evidence instead of an independent
    # trigger path.
    relation_with_scene = (
        relation_evidence
        and has_context
        and meaningful_scene_context
        and not queue_like
    )
    # High-confidence accident-response scene: some true multi-car accidents do
    # not show stable bbox contact, but they contain persistent people plus
    # response/road-control evidence around a small local static cluster. This
    # branch is intentionally strict to avoid returning to the v3.1 false-positive
    # pattern where road barriers alone triggered many normal highway scenes.
    scene_only_high_conf = (
        context_hits.get("person", 0) >= 6
        and (
            context_hits.get("construction vehicle", 0) >= 3
            or context_hits.get("traffic cone", 0) >= 3
            or context_hits.get("road barrier", 0) >= 6
        )
        and context_frames >= 6
        and 2 <= static_count <= 4
        and static_ratio <= 0.20
        and moving_count >= 7
        and not queue_like
        and 0.20 <= cluster_center_x_ratio <= 0.85
    )
    # Some true accident-response scenes are marked queue_like because many
    # vehicles continue passing around the incident. Allow this only when the
    # response evidence is much stronger than the normal scene-only branch.
    queue_scene_high_conf = (
        queue_like
        and context_hits.get("person", 0) >= 8
        and context_hits.get("construction vehicle", 0) >= 3
        and context_frames >= 8
        and 2 <= static_count <= 4
        and static_ratio <= 0.20
        and moving_count >= 10
        and 0.20 <= cluster_center_x_ratio <= 0.85
    )
    # Some real multi-car accidents are marked queue_like because the incident
    # itself creates local queuing behind the stopped vehicles. Allow only
    # strong relation evidence plus response-scene context through this gate;
    # do not use this for generic context_without_vehicle_relation cases.
    queue_contact_high_conf = (
        queue_like
        and relation_evidence
        and max_contact_frames >= 3
        and max_pair_stagnation_frames >= 8
        and context_frames >= 8
        and 4 <= static_count <= 6
        and static_ratio <= 0.26
        and moving_count >= 10
        and 0.18 <= cluster_center_x_ratio <= 0.85
        and (
            context_hits.get("person", 0) >= 6
            or (
                context_hits.get("person", 0) >= 3
                and context_hits.get("construction vehicle", 0) >= 3
            )
        )
    )
    response_static_pair = (
        static_pair_stagnation
        and has_context
        and static_count >= 2
        and static_count <= 4
        and static_ratio <= 0.22
        and moving_count >= 7
        and not queue_like
        and 0.20 <= cluster_center_x_ratio <= 0.85
        and (
            context_hits.get("person", 0) >= 2
            or (
                context_hits.get("person", 0) >= 1
                and (
                    context_hits.get("construction vehicle", 0) >= 3
                    or context_hits.get("traffic cone", 0) >= 3
                )
            )
        )
    )
    # No-person variant for clear roadway-core double-stopping cases. This is
    # deliberately narrower than response_static_pair to avoid converting
    # local congestion, single abnormal parking with a passing vehicle, or
    # two-wheel intrusion into multi-car accident evidence.
    long_core_static_pair = (
        static_pair_stagnation
        and max_pair_stagnation_frames >= 8
        and 2 <= static_count <= 3
        and static_ratio <= 0.16
        and moving_count >= 10
        and not queue_like
        and 0.25 <= cluster_center_x_ratio <= 0.75
        and context_hits.get("person", 0) == 0
        and context_hits.get("traffic cone", 0) < 3
        and context_hits.get("construction vehicle", 0) < 3
    )
    # Static-pair stagnation branch: targets accident-after-stopping cases
    # where two or more vehicles are already stopped in the roadway, but their
    # bboxes do not visibly touch.
    static_pair_stagnation_scene = response_static_pair or long_core_static_pair
    triggered = (
        relation_with_scene
        or scene_only_high_conf
        or queue_scene_high_conf
        or queue_contact_high_conf
        or static_pair_stagnation_scene
    )

    if not triggered:
        if has_context and not relation_evidence:
            proximity_reason = 'static_pair_without_scene' if static_pair_stagnation else 'context_without_vehicle_relation'
        elif relation_evidence and queue_like:
            proximity_reason = 'queue_like_suppressed'
        elif relation_evidence:
            proximity_reason = 'relation_without_scene_context'
        else:
            proximity_reason = None
    elif queue_contact_high_conf:
        proximity_reason = 'queue_contact_high_conf'
    elif static_pair_stagnation_scene and not relation_evidence:
        proximity_reason = 'static_pair_stagnation'

    return {
        "static_count": static_count,
        "moving_count": moving_count,
        "static_ratio": round(static_ratio, 3),
        "cluster_center_x_ratio": round(cluster_center_x_ratio, 3),
        "proximity": relation_evidence,
        "proximity_reason": proximity_reason,
        "contact_frames": max_contact_frames,
        "close_frames": max_close_frames,
        "pair_stagnation_frames": max_pair_stagnation_frames,
        "context_frames": context_frames,
        "context_hits": {k: v for k, v in sorted(context_hits.items()) if v > 0},
        "meaningful_scene_context": meaningful_scene_context,
        "scene_only_high_conf": scene_only_high_conf,
        "queue_scene_high_conf": queue_scene_high_conf,
        "queue_contact_high_conf": queue_contact_high_conf,
        "static_pair_stagnation": static_pair_stagnation,
        "static_pair_stagnation_scene": static_pair_stagnation_scene,
        "response_static_pair": response_static_pair,
        "long_core_static_pair": long_core_static_pair,
        "queue_like_frames": queue_like_frames,
        "checked_frames": checked_frames,
        "queue_like": queue_like,
        "triggered": triggered,
    }


def _compute_sarp(per_frame_boxes_by_label, img_width, img_height):
    """
    Surface Anomaly / Road-object Prompt module.

    SARP is intentionally conservative. It treats open-vocabulary debris labels
    as candidate generation only, then keeps candidates that lie in a broad road
    ROI and are not explained by vehicles, people, motorcycles, cones, barriers,
    or construction vehicles.
    """
    if not per_frame_boxes_by_label or img_width <= 0 or img_height <= 0:
        return {"ok": False, "reason": "no_frame_or_invalid_size"}

    roi = {
        "x": [0.03, 0.97],
        "y": [0.22, 0.98],
    }
    min_area_ratio = 0.000005
    max_area_ratio = 0.120
    evidence = []
    negative_boxes_by_frame = []
    known_labels = list(dict.fromkeys(TEXT_QUERIES + SARP_ENHANCED_QUERIES))
    label_hits = {k: 0 for k in SARP_TARGET_QUERIES}
    semantic_peak = {k: 0 for k in known_labels}
    rejected = {
        "outside_road_roi": 0,
        "area_out_of_range": 0,
        "negative_overlap": 0,
    }

    for frame_idx, boxes_by_label in enumerate(per_frame_boxes_by_label):
        for label in known_labels:
            semantic_peak[label] = max(semantic_peak[label], len(boxes_by_label.get(label, [])))

        negatives = []
        for label in SARP_NEGATIVE_QUERIES:
            negatives.extend(boxes_by_label.get(label, []))
        negative_boxes_by_frame.append({
            "frame": frame_idx,
            "boxes": [
                {
                    "label": neg_label,
                    "box": [round(float(x), 1) for x in neg_box],
                }
                for neg_label in sorted(SARP_NEGATIVE_QUERIES)
                for neg_box in boxes_by_label.get(neg_label, [])
            ],
        })

        for label in SARP_TARGET_QUERIES:
            for box in boxes_by_label.get(label, []):
                cx, cy = _box_center(box)
                if not (
                    roi["x"][0] * img_width <= cx <= roi["x"][1] * img_width
                    and roi["y"][0] * img_height <= cy <= roi["y"][1] * img_height
                ):
                    rejected["outside_road_roi"] += 1
                    continue

                area_ratio = _box_area(box) / max(1.0, float(img_width * img_height))
                if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                    rejected["area_out_of_range"] += 1
                    continue

                neg_overlap = False
                for neg in negatives:
                    neg_w = max(1.0, neg[2] - neg[0])
                    neg_h = max(1.0, neg[3] - neg[1])
                    expanded = _expand_box(neg, 0.10 * neg_w, 0.10 * neg_h, img_width, img_height)
                    if _box_iou(box, neg) >= 0.08 or _point_in_box((cx, cy), expanded):
                        neg_overlap = True
                        break
                if neg_overlap:
                    rejected["negative_overlap"] += 1
                    continue

                label_hits[label] += 1
                evidence.append({
                    "frame": frame_idx,
                    "label": label,
                    "box": [round(float(x), 1) for x in box],
                    "area_ratio": round(area_ratio, 6),
                    "center": [round(cx / img_width, 3), round(cy / img_height, 3)],
                })

    evidence_frames = len({item["frame"] for item in evidence})
    candidate_count = len(evidence)
    raw_triggered = evidence_frames >= 2 or candidate_count >= 3

    suppression_reason = None
    if raw_triggered:
        if semantic_peak.get("motorcycle", 0) >= 2:
            suppression_reason = "motorcycle_evidence"
        elif semantic_peak.get("traffic cone", 0) >= 5 or semantic_peak.get("construction vehicle", 0) >= 4:
            suppression_reason = "construction_evidence"

    triggered = raw_triggered and suppression_reason is None
    return {
        "ok": True,
        "triggered": triggered,
        "raw_triggered": raw_triggered,
        "primary_signal": "road_surface_object" if triggered else "none",
        "suppression_reason": suppression_reason,
        "candidate_count": candidate_count,
        "evidence_frames": evidence_frames,
        "label_hits": {k: v for k, v in sorted(label_hits.items()) if v > 0},
        "semantic_peak": {k: v for k, v in sorted(semantic_peak.items()) if v > 0},
        "roi": roi,
        "rejected": rejected,
        "examples": evidence[:8],
        "candidates": evidence,
        "negative_boxes_by_frame": negative_boxes_by_frame,
    }


def _detect_frame_ovd(frame, processor, model, device, box_threshold, text_threshold, queries=None):
    queries = list(queries or TEXT_QUERIES)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    text_input = ". ".join(queries)
    inputs = processor(
        images=rgb,
        text=text_input,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([rgb.shape[:2]], device=device)
    try:
        results = processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes
        )[0]
    except TypeError:
        results = processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            threshold=box_threshold,
            target_sizes=target_sizes
        )[0]

    label_list = results.get("text_labels") or [str(lb) for lb in results.get("labels", [])]
    boxes = results.get("boxes", [])

    frame_cnt = {q: 0 for q in queries}
    frame_vehicle_boxes = []
    frame_context_boxes = {q: [] for q in VSCM_CONTEXT_QUERIES}
    frame_boxes_by_label = {q: [] for q in queries}
    for idx_b, label in enumerate(label_list):
        label = str(label)
        box = None
        if idx_b < len(boxes):
            b = boxes[idx_b]
            box = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
        if label in frame_cnt:
            frame_cnt[label] += 1
        if label in frame_boxes_by_label and box is not None:
            frame_boxes_by_label[label].append(box)
        if label in VEHICLE_QUERIES and box is not None:
            frame_vehicle_boxes.append(box)
        if label in VSCM_CONTEXT_QUERIES and box is not None:
            frame_context_boxes[label].append(box)

    frame_vehicle_boxes = _dedupe_vehicle_boxes(frame_vehicle_boxes)
    return frame_cnt, frame_vehicle_boxes, frame_context_boxes, frame_boxes_by_label


def summarize_video(
    video_path,
    processor,
    model,
    device,
    box_threshold=0.30,
    text_threshold=0.25,
    frames=12,
    sarp_enhance=False,
    sarp_frames=24,
    sarp_box_threshold=0.22,
    sarp_text_threshold=0.20,
):
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    idxs = set(sample_indices(n, frames))
    sarp_idxs = set(sample_indices(n, sarp_frames)) if sarp_enhance else idxs
    detect_idxs = idxs | sarp_idxs

    labels_all = {q: 0 for q in TEXT_QUERIES}
    labels_peak = {q: 0 for q in TEXT_QUERIES}
    labels_first_frame = {q: None for q in TEXT_QUERIES}
    labels_last_frame  = {q: None for q in TEXT_QUERIES}

    # VSCM: per-frame list of boxes [x1,y1,x2,y2] (absolute pixels)
    per_frame_vehicle_boxes: list[list] = []
    per_frame_context_boxes: list[dict] = []
    per_frame_boxes_by_label: list[dict] = []
    per_frame_sarp_boxes_by_label: list[dict] = []

    used = 0
    fi = 0
    bad_reads = 0
    sample_seq = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            bad_reads += 1
            if bad_reads > 5:
                break
            fi += 1
            continue
        bad_reads = 0

        if fi in detect_idxs:
            frame_cnt = None
            frame_vehicle_boxes = None
            frame_context_boxes = None
            frame_boxes_by_label = None

            if fi in idxs:
                frame_cnt, frame_vehicle_boxes, frame_context_boxes, frame_boxes_by_label = _detect_frame_ovd(
                    frame, processor, model, device, box_threshold, text_threshold, queries=TEXT_QUERIES
                )

            if sarp_enhance and fi in sarp_idxs:
                _, _, _, sarp_boxes_by_label = _detect_frame_ovd(
                    frame,
                    processor,
                    model,
                    device,
                    sarp_box_threshold,
                    sarp_text_threshold,
                    queries=SARP_ENHANCED_QUERIES,
                )
                per_frame_sarp_boxes_by_label.append(sarp_boxes_by_label)
            elif not sarp_enhance and fi in idxs:
                per_frame_sarp_boxes_by_label.append(frame_boxes_by_label)

            if fi in idxs:
                if frame_cnt is None:
                    frame_cnt, frame_vehicle_boxes, frame_context_boxes, frame_boxes_by_label = _detect_frame_ovd(
                        frame, processor, model, device, box_threshold, text_threshold, queries=TEXT_QUERIES
                    )
                used += 1
                per_frame_vehicle_boxes.append(frame_vehicle_boxes)
                per_frame_context_boxes.append(frame_context_boxes)
                per_frame_boxes_by_label.append(frame_boxes_by_label)

                for k, v in frame_cnt.items():
                    labels_all[k] += v
                    if v > labels_peak[k]:
                        labels_peak[k] = v
                    if v > 0:
                        if labels_first_frame[k] is None:
                            labels_first_frame[k] = sample_seq
                        labels_last_frame[k] = sample_seq

                sample_seq += 1

        fi += 1

    cap.release()

    lcrm = _compute_lcrm_v1(per_frame_vehicle_boxes, img_width, img_height, labels_peak, labels_all)
    vscm_result = _compute_vscm_v3(per_frame_vehicle_boxes, per_frame_context_boxes, img_width, img_height)
    sarp = _compute_sarp(per_frame_sarp_boxes_by_label, img_width, img_height)
    sarp["enhanced"] = bool(sarp_enhance)
    sarp["sampled_frames"] = len(per_frame_sarp_boxes_by_label)
    if sarp_enhance:
        sarp["enhance_config"] = {
            "frames": sarp_frames,
            "box_threshold": sarp_box_threshold,
            "text_threshold": sarp_text_threshold,
            "query_count": len(SARP_ENHANCED_QUERIES),
        }
    vscm_debug = None
    if vscm_result and vscm_result.get("debug_only"):
        vscm_debug = vscm_result
        vscm = None
    else:
        vscm = vscm_result
    if vscm and vscm.get("triggered", False):
        motorcycle_peak = labels_peak.get("motorcycle", 0)
        motorcycle_sum = labels_all.get("motorcycle", 0)
        vscm["motorcycle_peak"] = motorcycle_peak
        vscm["motorcycle_sum"] = motorcycle_sum
        # VSCM is intended for multi-car accident evidence. If the same video
        # has any explicit motorcycle evidence, let the two-wheel OVD hint take
        # priority instead of injecting a competing vehicle-stagnation cue.
        if motorcycle_peak >= 1:
            vscm["triggered_before_motorcycle_suppression"] = True
            vscm["suppression_reason"] = "motorcycle_evidence"
            vscm["triggered"] = False

    return {
        "sum_count": labels_all,
        "peak_count": labels_peak,
        "first_frame": {k: v for k, v in labels_first_frame.items() if v is not None},
        "last_frame":  {k: v for k, v in labels_last_frame.items()  if v is not None},
        "sampled_frames": used,
        "lcrm": lcrm,
        "vscm": vscm,
        "vscm_debug": vscm_debug,
        "sarp": sarp,
    }

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-id", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--box-thres", type=float, default=0.30)
    ap.add_argument("--text-thres", type=float, default=0.25)
    ap.add_argument("--sarp-enhance", action="store_true", help="Run a SARP-only enhanced OVD branch.")
    ap.add_argument("--sarp-frames", type=int, default=24)
    ap.add_argument("--sarp-box-thres", type=float, default=0.22)
    ap.add_argument("--sarp-text-thres", type=float, default=0.20)
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = _Proc.from_pretrained(args.model_id, local_files_only=True)
    model = _Model.from_pretrained(args.model_id, local_files_only=True).to(device).eval()

    root = Path(args.video_root)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(args.jsonl, "r", encoding="utf-8") as f, open(outp, "w", encoding="utf-8") as w:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            video = obj.get("video", "")
            vp = root / video
            rec = {"video": video, "ok": False, "ovd": None}
            if vp.exists():
                try:
                    rec["ovd"] = summarize_video(
                        vp, processor, model, device,
                        box_threshold=args.box_thres,
                        text_threshold=args.text_thres,
                        frames=args.frames,
                        sarp_enhance=args.sarp_enhance,
                        sarp_frames=args.sarp_frames,
                        sarp_box_threshold=args.sarp_box_thres,
                        sarp_text_threshold=args.sarp_text_thres,
                    )
                    rec["ok"] = True
                except Exception as e:
                    rec["err"] = str(e)
            else:
                rec["err"] = "video_not_found"

            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += 1
            if args.max_samples > 0 and total >= args.max_samples:
                break

    print("saved:", outp, "records:", total)

if __name__ == "__main__":
    main()
