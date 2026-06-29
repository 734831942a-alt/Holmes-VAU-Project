import json
from pathlib import Path

TEST_JSONL = Path("/root/autodl-tmp/高架桥数据/traffic_test.jsonl")
OVD_JSONL = Path("/root/autodl-tmp/高架桥数据/traffic_test_ovd_summary.jsonl")
OUT_JSONL = Path("/root/autodl-tmp/高架桥数据/traffic_test_with_ovd_prompt.jsonl")

def load_ovd_map(path):
    m = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        m[o.get("video", "")] = o
    return m

def make_hint(ovd_obj):
    if not ovd_obj or (not ovd_obj.get("ok")) or (not ovd_obj.get("ovd")):
        return "Perception hint: unavailable."
    s = ovd_obj["ovd"]["sum_count"]
    p = ovd_obj["ovd"]["peak_count"]

    # Low-noise, task-oriented signals
    has_motor = p.get("motorcycle", 0) > 0
    heavy_peak = p.get("truck", 0) + p.get("bus", 0)
    obstacle_peak = max(
        p.get("debris", 0),
        p.get("road obstacle", 0),
        p.get("scattered object", 0),
        p.get("road barrier", 0),
        p.get("traffic cone", 0)
    )
    traffic_density = p.get("car", 0) + p.get("truck", 0) + p.get("bus", 0)

    lines = [
        f"Perception hint: motorcycle_present={int(has_motor)}",
        f"heavy_vehicle_peak={heavy_peak}",
        f"road_obstacle_peak={obstacle_peak}",
        f"traffic_density_peak={traffic_density}",
    ]
    return "; ".join(lines) + "."

def main():
    ovd_map = load_ovd_map(OVD_JSONL)
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with OUT_JSONL.open("w", encoding="utf-8") as w:
        for line in TEST_JSONL.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            video = obj.get("video", "")
            ovd = ovd_map.get(video)
            hint = make_hint(ovd)

            if "conversations" in obj and obj["conversations"]:
                old = obj["conversations"][0].get("value", "")
                obj["conversations"][0]["value"] = hint + "\n" + old

            obj["ovd_ok"] = bool(ovd and ovd.get("ok"))
            obj["ovd_summary"] = ovd.get("ovd") if ovd else None
            if ovd and "err" in ovd:
                obj["ovd_err"] = ovd["err"]

            w.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("saved:", OUT_JSONL)

if __name__ == "__main__":
    main()
