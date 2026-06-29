"""
build_ovd_prompt_zh.py

为训练集和测试集构建中文异常目标感知 hint（E3/E4 实验使用）。
将 OVD 检测摘要转换为面向异常类别的中文描述，注入到 conversations[0] 头部。

用法（服务器）：
  # 构建测试集（E3 推理用）
  python build_ovd_prompt_zh.py \
      --input  /root/autodl-tmp/高架桥数据/traffic_test.jsonl \
      --ovd    /root/autodl-tmp/高架桥数据/traffic_test_ovd_summary.jsonl \
      --output /root/autodl-tmp/高架桥数据/traffic_test_ovd_zh.jsonl

  # 构建训练集（E4 训练用）
  python build_ovd_prompt_zh.py \
      --input  /root/autodl-tmp/高架桥数据/traffic_train.jsonl \
      --ovd    /root/autodl-tmp/高架桥数据/traffic_train_ovd_summary.jsonl \
      --output /root/autodl-tmp/高架桥数据/traffic_train_ovd_zh.jsonl

hint 设计原则（对标6类标签）：
  - 摩托车/电动车 → 对应"二轮车辆闯入"
  - 路面障碍物/锥桶 → 对应"占道施工"/"抛洒物"
  - 重型车辆 → 对应"多车事故"辅助
  - 车辆密度 → 对应"拥堵"辅助
  - 静止车辆 → 对应"异常停车"辅助
  只保留 >0 的信号，零值不输出，避免干扰。
"""

import argparse
import json
from pathlib import Path


def load_ovd_map(ovd_path: Path) -> dict:
    m = {}
    for line in ovd_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        key = obj.get("video", "")
        m[key] = obj
    return m



def make_zh_hint(ovd_obj: dict) -> str:
    """
    Convert OVD/VSCM summary to a conservative Chinese hint.

    VSCM v3 is weak evidence. The hint may mention nearby scene evidence,
    but it must not force the model toward the multi-car-accident label.
    """
    if not ovd_obj or not ovd_obj.get("ok"):
        return ""

    ovd = ovd_obj.get("ovd")
    if not ovd:
        return ""

    vscm = ovd.get("vscm")
    if not vscm or not vscm.get("triggered", False):
        return ""

    static_count = vscm["static_count"]
    moving_count = vscm["moving_count"]
    reason = vscm.get("proximity_reason")
    pair_frames = vscm.get("pair_stagnation_frames", 0)
    context_hits = vscm.get("context_hits") or {}
    context_text = ""
    if context_hits:
        zh = {
            "person": "\u4eba\u5458",
            "traffic cone": "\u4ea4\u901a\u9525\u6876",
            "construction vehicle": "\u5de5\u7a0b\u8f66\u8f86",
            "road barrier": "\u8def\u969c/\u62a4\u680f",
        }
        parts = [f"{zh.get(k, k)}{v}\u5e27\u6b21" for k, v in context_hits.items()]
        context_text = "\uff0c\u9644\u8fd1\u8fd8\u68c0\u6d4b\u5230" + "\u3001".join(parts)

    if reason in {"static_pair_stagnation", "queue_contact_high_conf"}:
        return (
            f"[事故后双车滞留线索] 采样帧中检测到两辆及以上机动车在局部区域持续静止"
            f"（最长持续约{pair_frames}个采样帧），周围仍有{moving_count}辆车通行{context_text}。"
            f"该线索仅表示可能存在事故后滞留、临时停车或局部阻塞；请结合视频画面独立判断事件类型，"
            f"不要仅凭该线索确定为多车事故。"
        )

    return (
        f"[局部车辆静止线索] 采样帧中检测到{static_count}辆车在局部区域持续静止，"
        f"周围仍有{moving_count}辆车通行{context_text}。"
        f"该线索仅表示局部交通状态异常，可能对应事故后滞留、临时停车或局部阻塞；"
        f"请以视频画面为主独立判断事件类型，不要仅凭该线索确定类别。"
    )

def process(input_path: Path, ovd_path: Path, output_path: Path) -> None:
    ovd_map = load_ovd_map(ovd_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    injected = 0
    skipped = 0
    # 诊断：各关键词触发次数
    signal_counts: dict = {
        "\u5c40\u90e8\u8f66\u8f86\u9759\u6b62\u7ebf\u7d22": 0,
    }

    with output_path.open("w", encoding="utf-8") as w:
        for line in input_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            video = obj.get("video", "")
            ovd_obj = ovd_map.get(video)
            hint = make_zh_hint(ovd_obj) if ovd_obj else ""

            if hint:
                # 注入到 conversations[0] 的 human turn 头部
                if "conversations" in obj and obj["conversations"]:
                    old_val = obj["conversations"][0].get("value", "")
                    obj["conversations"][0]["value"] = hint + "\n" + old_val
                # 同时维护 prompt 字段（若存在）
                if "prompt" in obj and obj["prompt"]:
                    obj["prompt"] = hint + "\n" + obj["prompt"]
                obj["ovd_zh_injected"] = True
                injected += 1
                for k in signal_counts:
                    if k in hint:
                        signal_counts[k] += 1
            else:
                obj["ovd_zh_injected"] = False
                skipped += 1

            w.write(json.dumps(obj, ensure_ascii=False) + "\n")
            total += 1

    print(f"Done. total={total}, injected={injected} ({injected/max(total,1)*100:.1f}%), skipped={skipped}")
    print(f"Output: {output_path}")
    print(f"\n--- 各信号触发统计（E7 VSCM）---")
    for k, v in signal_counts.items():
        print(f"  {k}: {v} 条 ({v/max(total,1)*100:.1f}%)")

    # 打印一条注入了 hint 的示例
    for line in output_path.read_text(encoding="utf-8").splitlines():
        first = json.loads(line)
        if first.get("ovd_zh_injected") and "conversations" in first:
            val = first["conversations"][0].get("value", "")
            print("\n--- 示例 hint ---")
            print(val.split("\n")[0])
            break


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, help="原始 jsonl（train 或 test）")
    p.add_argument("--ovd",    required=True, help="OVD 摘要 jsonl（traffic_*_ovd_summary.jsonl）")
    p.add_argument("--output", required=True, help="输出 jsonl")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process(Path(args.input), Path(args.ovd), Path(args.output))
