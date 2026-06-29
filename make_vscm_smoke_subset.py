"""
make_vscm_smoke_subset.py — 按类别采样生成 VSCM 烟雾测试用 GT subset。

用法（服务器）：
    python make_vscm_smoke_subset.py \
        --gt    /root/autodl-tmp/高架桥数据/traffic_train.jsonl \
        --out   /root/autodl-tmp/高架桥数据/vscm_smoke_gt.jsonl

输出：40 条（多车事故×20，拥堵×10，异常停车×10），供 ovd_video_summary.py 处理。

完整 smoke 流程：
    # 步骤1：生成 GT subset（本脚本，秒级）
    python make_vscm_smoke_subset.py \
        --gt  /root/autodl-tmp/高架桥数据/traffic_train.jsonl \
        --out /root/autodl-tmp/高架桥数据/vscm_smoke_gt.jsonl

    # 步骤2：跑 OVD（~13 分钟，40 条）
    python ovd_video_summary.py \
        --jsonl      /root/autodl-tmp/高架桥数据/vscm_smoke_gt.jsonl \
        --video-root /root/autodl-tmp/高架桥数据/video \
        --out        /root/autodl-tmp/高架桥数据/vscm_smoke_ovd.jsonl

    # 步骤3：审计触发率
    python audit_vscm.py \
        --gt  /root/autodl-tmp/高架桥数据/vscm_smoke_gt.jsonl \
        --ovd /root/autodl-tmp/高架桥数据/vscm_smoke_ovd.jsonl \
        --targets 多车事故,拥堵,异常停车

验收标准（覆盖内触发率）：
    多车事故  >= 30%
    拥堵      <= 10%
    异常停车  <= 10%
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


SAMPLE_PLAN = {
    "多车事故":     60,
    "拥堵":         30,
    "异常停车":     30,
    "占道施工":     30,
    "二轮车辆闯入": 30,
    "抛洒物":       30,
}


def extract_label(video: str) -> str:
    return video.split("_")[0] if video else ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt",   required=True, help="完整训练集 jsonl")
    p.add_argument("--out",  required=True, help="输出的 smoke subset jsonl")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)

    # 按类别收集所有行
    buckets: dict[str, list] = defaultdict(list)
    for line in Path(args.gt).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        label = extract_label(obj.get("video", ""))
        if label in SAMPLE_PLAN:
            buckets[label].append(obj)

    # 打印可用数量
    for label, n in SAMPLE_PLAN.items():
        avail = len(buckets[label])
        print(f"  {label}: 可用 {avail} 条，采样 {min(n, avail)} 条")

    # 采样并写出
    rows = []
    for label, n in SAMPLE_PLAN.items():
        pool = buckets[label]
        picked = random.sample(pool, min(n, len(pool)))
        rows.extend(picked)

    # 打乱顺序（让三类交错分布，避免 OVD 连续处理同一类）
    random.shuffle(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n输出: {out}  共 {len(rows)} 条")
    print("下一步: python ovd_video_summary.py --jsonl <out> --video-root <video_root> --out vscm_smoke_ovd.jsonl")


if __name__ == "__main__":
    main()
