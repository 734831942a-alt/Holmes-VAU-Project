"""
audit_vscm.py — 审计收紧版 VSCM 对各类别的触发率。

用法（服务器上跑）：
    python audit_vscm.py \
        --gt      /root/autodl-tmp/高架桥数据/traffic_train.jsonl \
        --ovd     /root/autodl-tmp/高架桥数据/traffic_train_ovd_summary_v5.jsonl \
        --targets 多车事故,拥堵,异常停车,占道施工,二轮车辆闯入,抛洒物

验收标准：
    多车事故  触发率 >= 30%
    拥堵      误触发率 <= 10%
    异常停车  误触发率 <= 10%
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt",  required=True, help="原始 jsonl（含 gt label）")
    p.add_argument("--ovd", required=True, help="OVD summary jsonl（含 vscm 字段）")
    p.add_argument("--targets", default="多车事故,拥堵,异常停车,占道施工,二轮车辆闯入,抛洒物")
    return p.parse_args()


def extract_gt_label(obj: dict) -> str:
    """从视频文件名前缀提取类别标签，如 '多车事故_20231201...' → '多车事故'。"""
    video = obj.get("video", "")
    return video.split("_")[0] if video else ""


def main():
    args = parse_args()
    targets = [t.strip() for t in args.targets.split(",")]

    # 加载 OVD summary：video -> vscm
    ovd_map: dict[str, dict] = {}
    for line in Path(args.ovd).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        vscm = o.get("ovd", {})
        if isinstance(vscm, dict):
            vscm = vscm.get("vscm")
        else:
            vscm = None
        ovd_map[o.get("video", "")] = vscm

    # 统计：per label -> {total, triggered, no_vscm}
    stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "triggered": 0, "no_ovd": 0})

    for line in Path(args.gt).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        label = extract_gt_label(obj)
        video = obj.get("video", "")

        matched = next((t for t in targets if t in label), None)
        if matched is None:
            continue

        stats[matched]["total"] += 1
        vscm = ovd_map.get(video)
        if vscm is None:
            stats[matched]["no_ovd"] += 1
            continue
        if vscm.get("triggered", False):
            stats[matched]["triggered"] += 1

    print(f"\n{'类别':<12} {'总数':>6} {'有OVD':>6} {'触发':>6} {'全量触发率':>10} {'覆盖内触发率':>12}")
    print("-" * 58)
    for label in targets:
        s = stats[label]
        total  = s["total"]
        trig   = s["triggered"]
        no_ovd = s["no_ovd"]
        covered = total - no_ovd
        rate_all     = trig / total   * 100 if total   > 0 else 0.0
        rate_covered = trig / covered * 100 if covered > 0 else 0.0
        flag = ""
        if label == "多车事故" and rate_covered < 30:
            flag = " ← 未达标(≥30%)"
        elif label in ("拥堵", "异常停车") and rate_covered > 10:
            flag = " ← 误触发过高(≤10%)"
        print(f"{label:<12} {total:>6} {covered:>6} {trig:>6} {rate_all:>9.1f}% {rate_covered:>11.1f}%{flag}")

    print("\n验收标准（覆盖内触发率）：多车事故≥30%，拥堵/异常停车≤10%")


if __name__ == "__main__":
    main()
