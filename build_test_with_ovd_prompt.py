import json
from pathlib import Path

TEST_JSONL = Path("/root/autodl-tmp/高架桥数据/traffic_test.jsonl")
OVD_JSONL = Path("/root/autodl-tmp/高架桥数据/traffic_test_ovd_summary_v7.jsonl")
OUT_JSONL = Path("/root/autodl-tmp/高架桥数据/traffic_test_with_ovd_prompt_v8.jsonl")

def load_ovd_map(path):
    m = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        m[o.get("video", "")] = o
    return m


def make_hint(ovd_obj):
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

def main():
    ovd_map = load_ovd_map(OVD_JSONL)
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    injected = 0
    total = 0
    with OUT_JSONL.open("w", encoding="utf-8") as w:
        for line in TEST_JSONL.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            video = obj.get("video", "")
            ovd = ovd_map.get(video)
            hint = make_hint(ovd)

            if hint:
                if "conversations" in obj and obj["conversations"]:
                    old = obj["conversations"][0].get("value", "")
                    obj["conversations"][0]["value"] = hint + "\n" + old
                injected += 1

            obj["ovd_ok"] = bool(ovd and ovd.get("ok"))
            obj["ovd_summary"] = ovd.get("ovd") if ovd else None
            if ovd and "err" in ovd:
                obj["ovd_err"] = ovd["err"]

            w.write(json.dumps(obj, ensure_ascii=False) + "\n")
            total += 1

    print(f"saved: {OUT_JSONL}")
    print(f"total={total}, injected={injected} ({injected/max(total,1)*100:.1f}%)")

if __name__ == "__main__":
    main()
