# Frozen transplant from:
# - ovd_video_summary.py
# - eval_traffic_fixed.py
# - build_ovd_prompt_zh.py
# Source commit: 8b98e6cda09d078061eb308d78feec75fa23e6fd
"""Canonical vocabulary for traffic anomaly classification and OVD evidence."""

from typing import Dict, List, Tuple

LABELS: List[str] = [
    "多车事故",
    "拥堵",
    "异常停车",
    "占道施工",
    "二轮车辆闯入",
    "抛洒物",
]

ALIASES: Dict[str, List[str]] = {
    "多车事故": ["多车事故", "交通事故", "多车", "碰撞", "追尾"],
    "拥堵": ["拥堵", "交通堵塞", "堵塞", "缓行", "排队"],
    "异常停车": ["异常停车", "停车", "停靠", "静止"],
    "占道施工": ["占道施工", "施工", "锥桶", "施工车", "占道"],
    "二轮车辆闯入": ["二轮车辆闯入", "二轮闯入", "二轮", "非机动车闯入", "摩托车", "电动车"],
    "抛洒物": ["抛洒物", "抛撒物", "散落物", "异物"],
}

# Base OVD queries from ovd_video_summary.py.
TEXT_QUERIES: List[str] = [
    "car", "truck", "bus", "motorcycle", "person",
    "traffic cone", "construction vehicle", "road barrier",
    "debris", "road obstacle", "scattered object",
]
VEHICLE_QUERIES = {"car", "truck", "bus"}
VSCM_CONTEXT_QUERIES = {"person", "traffic cone", "construction vehicle", "road barrier"}

# SARP extends the base list. The extra terms are intentionally retained rather
# than silently reconciled; see issues/01-vocab-conflict.md.
SARP_TARGET_QUERIES = {
    "debris", "road debris", "road obstacle", "scattered object",
    "road spill", "trash",
}
SARP_NEGATIVE_QUERIES = {
    "car", "truck", "bus", "motorcycle", "person",
    "traffic cone", "construction vehicle", "road barrier",
}
SARP_ENHANCED_QUERIES = sorted(SARP_NEGATIVE_QUERIES | SARP_TARGET_QUERIES)

# The source semantic notes are many-to-many, not a strict class detector:
# road obstacles/cones can support construction or debris, while generic
# vehicle queries can support accident, congestion, or parking. All source
# possibilities are preserved instead of choosing one.
CLASS_ENGLISH_QUERIES: Dict[str, Tuple[str, ...]] = {
    "多车事故": ("car", "truck", "bus"),
    "拥堵": ("car", "truck", "bus"),
    "异常停车": ("car", "truck", "bus"),
    "占道施工": ("traffic cone", "construction vehicle", "road barrier", "road obstacle"),
    "二轮车辆闯入": ("motorcycle",),
    "抛洒物": ("debris", "road debris", "road obstacle", "scattered object", "road spill", "trash"),
}

SEMANTIC_HINTS: Dict[str, str] = {
    "多车事故": "重型车辆或多车辆信号只作为多车事故辅助证据。",
    "拥堵": "车辆密度信号只作为拥堵辅助证据。",
    "异常停车": "静止车辆信号只作为异常停车辅助证据。",
    "占道施工": "路面障碍物、锥桶、施工车辆或路障可作为占道施工线索。",
    "二轮车辆闯入": "摩托车或电动车信号对应二轮车辆闯入线索。",
    "抛洒物": "路面障碍物、碎屑、散落物、溢洒物或垃圾可作为抛洒物线索。",
}

ALL_OVD_QUERIES: Tuple[str, ...] = tuple(
    dict.fromkeys(TEXT_QUERIES + SARP_ENHANCED_QUERIES)
)

__all__ = [
    "LABELS", "ALIASES", "TEXT_QUERIES", "VEHICLE_QUERIES",
    "VSCM_CONTEXT_QUERIES", "SARP_TARGET_QUERIES",
    "SARP_NEGATIVE_QUERIES", "SARP_ENHANCED_QUERIES",
    "CLASS_ENGLISH_QUERIES", "SEMANTIC_HINTS", "ALL_OVD_QUERIES",
]
