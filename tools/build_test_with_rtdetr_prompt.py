import json
from pathlib import Path

TEST_JSONL = Path("/root/autodl-tmp/高架桥数据/traffic_test.jsonl")
DET_JSONL = Path("/root/autodl-tmp/高架桥数据/traffic_test_rtdetr_summary.jsonl")
OUT_JSONL = Path("/root/autodl-tmp/高架桥数据/traffic_test_with_rtdetr_prompt.jsonl")

def load_det_map(path):
    m = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        m[obj.get("video", "")] = obj
    return m

def det_text(det_obj):
    if not det_obj or (not det_obj.get("ok")) or (not det_obj.get("det")):
        return "Detection summary: unavailable."
    d = det_obj["det"]
    mean_count = d.get("mean_count", {})
    max_count = d.get("max_count", {})
    return (
        "Detection summary: "
        f"mean(person={mean_count.get('person',0)}, car={mean_count.get('car',0)}, "
        f"motorcycle={mean_count.get('motorcycle',0)}, bus={mean_count.get('bus',0)}, "
        f"truck={mean_count.get('truck',0)}); "
        f"max(person={max_count.get('person',0)}, car={max_count.get('car',0)}, "
        f"motorcycle={max_count.get('motorcycle',0)}, bus={max_count.get('bus',0)}, "
        f"truck={max_count.get('truck',0)})."
    )

def main():
    det_map = load_det_map(DET_JSONL)
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with OUT_JSONL.open("w", encoding="utf-8") as w:
        for line in TEST_JSONL.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            video = obj.get("video", "")
            det = det_map.get(video)
            hint = det_text(det)

            if "conversations" in obj and obj["conversations"]:
                old = obj["conversations"][0].get("value", "")
                obj["conversations"][0]["value"] = hint + "\n" + old

            obj["det_ok"] = bool(det and det.get("ok"))
            obj["det_summary"] = det.get("det") if det else None
            if det and "err" in det:
                obj["det_err"] = det["err"]

            w.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("saved:", OUT_JSONL)

if __name__ == "__main__":
    main()
