"""
inspect_vscm_samples.py — 从 smoke 结果里找典型正/负样本，供人工核查。

用法：
    python inspect_vscm_samples.py \
        --gt  /root/autodl-tmp/高架桥数据/vscm_smoke_gt.jsonl \
        --ovd /root/autodl-tmp/高架桥数据/vscm_smoke_ovd.jsonl \
        --video-root /root/autodl-tmp/高架桥数据/video

输出 6 类样本（各 1 条）：
    [多车事故] VSCM 正确触发
    [多车事故] VSCM 未触发（漏检）
    [拥堵]     VSCM 误触发
    [拥堵]     VSCM 未触发（正确）
    [异常停车] VSCM 误触发
    [异常停车] VSCM 未触发（正确）
"""
import argparse
import json
from pathlib import Path


def extract_label(video: str) -> str:
    return video.split("_")[0] if video else ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt",         required=True)
    p.add_argument("--ovd",        required=True)
    p.add_argument("--video-root", required=True)
    args = p.parse_args()

    video_root = Path(args.video_root)

    # 加载 OVD summary
    ovd_map: dict[str, dict] = {}
    for line in Path(args.ovd).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        ovd_map[o.get("video", "")] = o

    # 定义要找的 6 类槽位
    slots = {
        "多车事故_triggered":    {"label": "多车事故",  "triggered": True,  "found": None},
        "多车事故_not_triggered": {"label": "多车事故",  "triggered": False, "found": None},
        "拥堵_triggered":        {"label": "拥堵",      "triggered": True,  "found": None},
        "拥堵_not_triggered":    {"label": "拥堵",      "triggered": False, "found": None},
        "异常停车_triggered":    {"label": "异常停车",  "triggered": True,  "found": None},
        "异常停车_not_triggered":{"label": "异常停车",  "triggered": False, "found": None},
    }

    for line in Path(args.gt).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        video = obj.get("video", "")
        label = extract_label(video)

        ovd_rec = ovd_map.get(video)
        if not ovd_rec or not ovd_rec.get("ok"):
            continue

        vscm = ovd_rec.get("ovd", {})
        if isinstance(vscm, dict):
            vscm = vscm.get("vscm")
        if vscm is None:
            triggered = False
        else:
            triggered = bool(vscm.get("triggered", False))

        for key, slot in slots.items():
            if slot["found"] is not None:
                continue
            if slot["label"] == label and slot["triggered"] == triggered:
                slot["found"] = {
                    "video": video,
                    "path":  str(video_root / video),
                    "vscm":  vscm,
                }

        if all(s["found"] is not None for s in slots.values()):
            break

    # 打印结果
    LABELS = {
        "多车事故_triggered":     "[多车事故] ✅ VSCM 正确触发",
        "多车事故_not_triggered": "[多车事故] ❌ VSCM 未触发（漏检）",
        "拥堵_triggered":         "[拥堵]     ⚠️  VSCM 误触发",
        "拥堵_not_triggered":     "[拥堵]     ✅ VSCM 未触发（正确）",
        "异常停车_triggered":     "[异常停车] ⚠️  VSCM 误触发",
        "异常停车_not_triggered": "[异常停车] ✅ VSCM 未触发（正确）",
    }

    print()
    for key, slot in slots.items():
        print(LABELS[key])
        if slot["found"] is None:
            print("  ※ smoke 集中未找到此类样本")
        else:
            f = slot["found"]
            print(f"  视频: {f['path']}")
            if f["vscm"]:
                v = f["vscm"]
                print(f"  VSCM: static={v.get('static_count')}  moving={v.get('moving_count')}  "
                      f"ratio={v.get('static_ratio')}  reason={v.get('proximity_reason', '-')}")
            else:
                print("  VSCM: None（OVD 基础条件未满足）")
        print()


if __name__ == "__main__":
    main()
