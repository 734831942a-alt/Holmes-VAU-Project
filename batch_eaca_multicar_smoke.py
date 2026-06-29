import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


TARGET_LABELS = ["多车事故", "拥堵", "异常停车", "占道施工", "二轮车辆闯入", "抛洒物"]
DEFAULT_PLAN = {
    "多车事故": 48,
    "拥堵": 60,
    "异常停车": 60,
    "占道施工": 20,
    "二轮车辆闯入": 20,
    "抛洒物": 20,
}


EACA_PROMPT_V2_SOFT = """你是高速交通事件证据核查器。只根据这段短视频片段回答，不要沿用先验类别。

请严格按下面格式输出，每行只写 是 或 否。不要输出 LABEL，不要解释。

ACCIDENT: 是否清楚看到两辆及以上机动车发生碰撞、追尾、剐蹭、事故后近距离共同滞留，或事故处置现场证据？
PARKING: 是否主要是一辆机动车或少数相隔较远车辆静止/停靠/故障停车？
CONGESTION: 是否主要是大范围车辆排队、密集缓行、整体低速或走走停停？
CONSTRUCTION: 是否看到锥桶、围挡、施工车辆、施工导流或占道作业？
TWOWHEEL: 是否看到摩托车、电动车、非机动车进入主路或异常行驶？
DEBRIS: 是否看到路面抛洒物、散落物、异物、障碍物？

注意：
- 没有明确碰撞或事故后近距离共同滞留证据时，ACCIDENT 写 否。
- 多辆车低速排队不是多车事故，优先算 CONGESTION。
- 单车或相隔较远车辆静止不是多车事故，优先算 PARKING。
"""


EACA_PROMPT_V3_EVIDENCE = """你是高速交通事件证据核查器。只根据这段短视频片段回答，不要沿用先验类别。

请严格按下面格式输出，每行只写 是 或 否。不要输出类别名称，不要解释。

COLLISION: 是否清楚看到两辆及以上机动车发生碰撞、追尾、剐蹭，或看到事故处置现场证据？
CLOSE_MULTI_STOP: 是否看到两辆及以上机动车在同一局部区域近距离共同静止/滞留，像事故后聚集？
PARKING: 是否主要是一辆机动车，或少数相隔较远车辆静止/停靠/故障停车？
CONGESTION: 是否主要是大范围车辆排队、密集缓行、整体低速或走走停停？
CONSTRUCTION: 是否看到锥桶、围挡、施工车辆、施工导流或占道作业？
TWOWHEEL: 是否看到摩托车、电动车、非机动车进入主路或异常行驶？
DEBRIS: 是否看到路面抛洒物、散落物、异物、障碍物？

注意：
- 没有明确碰撞/追尾/事故处置证据时，COLLISION 写 否。
- 多辆车只是排队缓行时，CLOSE_MULTI_STOP 写 否，CONGESTION 写 是。
- 多辆车低速排队不是多车事故，优先算 CONGESTION。
- 单车或相隔较远车辆静止不是多车事故，优先算 PARKING。
- 如果只是看到多辆车同框，但没有碰撞/追尾/近距离共同滞留证据，COLLISION 和 CLOSE_MULTI_STOP 都必须写 否。
"""


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


def label_from_video(video):
    return str(video or "").split("_")[0]


def make_subset(rows, plan, max_samples):
    buckets = defaultdict(list)
    for row in rows:
        label = label_from_video(row.get("video", ""))
        if label in plan:
            buckets[label].append(row)

    picked = []
    for label, n in plan.items():
        picked.extend(buckets[label][: min(n, len(buckets[label]))])
    if max_samples > 0:
        picked = picked[:max_samples]
    return picked


def resolve_video_path(video_root, video_item):
    p = Path(str(video_item))
    if p.is_absolute():
        return str(p)
    return str(Path(video_root) / str(video_item))


def load_video_root(meta_path):
    with Path(meta_path).open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return next(iter(meta.values()))["root"]


def clip_frame_indices(total_frames, clip_id, num_clips, frames_per_clip):
    if total_frames <= 0:
        return []
    start = int(math.floor(total_frames * clip_id / num_clips))
    end = int(math.floor(total_frames * (clip_id + 1) / num_clips)) - 1
    start = max(0, min(start, total_frames - 1))
    end = max(start, min(end, total_frames - 1))
    if frames_per_clip <= 1 or start == end:
        return [int((start + end) / 2)]
    return [int(round(start + (end - start) * i / (frames_per_clip - 1))) for i in range(frames_per_clip)]


def get_pixel_values(vr, frame_indices, input_size=448, max_num=1):
    import torch
    from PIL import Image

    from holmesvau.internvl_utils import build_transform, dynamic_preprocess

    transform = build_transform(input_size=input_size)
    pixel_values_list = []
    num_patches_list = []
    for idx in frame_indices:
        img = Image.fromarray(vr[idx].asnumpy()).convert("RGB")
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = torch.stack([transform(tile) for tile in tiles])
        pixel_values_list.append(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
    return torch.cat(pixel_values_list), num_patches_list


def chat_clip(vr, frame_indices, prompt, model, tokenizer, generation_config):
    import torch

    pixel_values, num_patches_list = get_pixel_values(vr, frame_indices)
    pixel_values = pixel_values.to(torch.bfloat16).to(model.device)
    video_prefix = "".join([f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))])
    question = video_prefix + prompt
    response, _ = model.chat(
        tokenizer,
        pixel_values,
        question,
        generation_config,
        num_patches_list=num_patches_list,
        history=None,
        return_history=True,
    )
    return "" if response is None else str(response).strip()


FIELD_ALIASES = {
    "COLLISION": ["COLLISION", "碰撞证据", "碰撞", "追尾"],
    "CLOSE_MULTI_STOP": ["CLOSE_MULTI_STOP", "近距离多车滞留", "多车滞留", "共同静止", "共同滞留"],
    "ACCIDENT": ["ACCIDENT", "事故证据", "事故"],
    "PARKING": ["PARKING", "停车", "异常停车", "单车停车"],
    "CONGESTION": ["CONGESTION", "拥堵", "全局拥堵"],
    "CONSTRUCTION": ["CONSTRUCTION", "施工", "占道施工"],
    "TWOWHEEL": ["TWOWHEEL", "二轮", "二轮车", "非机动车"],
    "DEBRIS": ["DEBRIS", "抛洒物", "散落物", "异物"],
}


def parse_yesno(text, field):
    aliases = FIELD_ALIASES.get(field, [field])
    for name in aliases:
        m = re.search(rf"{re.escape(name)}\s*[:：=]\s*(是|否)", text, flags=re.I)
        if m:
            return m.group(1) == "是"
    return None


def parse_label(text):
    m = re.search(r"LABEL\s*[:：]\s*([^\n\r,，。；; ]+)", text, flags=re.I)
    if m:
        cand = m.group(1).strip()
        for label in TARGET_LABELS:
            if label in cand:
                return label
    # Fallback: use the last explicit label mention.
    best_pos = -1
    best_label = None
    for label in TARGET_LABELS:
        pos = text.rfind(label)
        if pos > best_pos:
            best_pos = pos
            best_label = label
    return best_label


def parse_clip_response(text, eaca_version="v2_soft"):
    out = {
        "collision": parse_yesno(text, "COLLISION"),
        "close_multi_stop": parse_yesno(text, "CLOSE_MULTI_STOP"),
        "accident": parse_yesno(text, "ACCIDENT"),
        "parking": parse_yesno(text, "PARKING"),
        "congestion": parse_yesno(text, "CONGESTION"),
        "construction": parse_yesno(text, "CONSTRUCTION"),
        "twowheel": parse_yesno(text, "TWOWHEEL"),
        "debris": parse_yesno(text, "DEBRIS"),
        "label": parse_label(text) if eaca_version == "v2_soft" else None,
    }
    for key in ["collision", "close_multi_stop", "accident", "parking", "congestion", "construction", "twowheel", "debris"]:
        if out[key] is None:
            out[key] = False
    if eaca_version == "v3_evidence":
        out["accident"] = bool(out["accident"] or out["collision"] or out["close_multi_stop"])
    return out


def aggregate_clips(clips, min_accident_clips=2, eaca_version="v2_soft"):
    counts = Counter()
    label_counts = Counter()
    for clip in clips:
        parsed = clip["parsed"]
        for key in ["collision", "close_multi_stop", "accident", "parking", "congestion", "construction", "twowheel", "debris"]:
            if parsed.get(key):
                counts[key] += 1
        if parsed.get("label"):
            label_counts[parsed["label"]] += 1

    n = max(1, len(clips))

    if counts["twowheel"] >= 1:
        label = "二轮车辆闯入"
        reason = "twowheel_clip_evidence"
    elif counts["construction"] >= 1:
        label = "占道施工"
        reason = "construction_clip_evidence"
    elif counts["debris"] >= 1:
        label = "抛洒物"
        reason = "debris_clip_evidence"
    elif eaca_version == "v3_evidence" and counts["collision"] >= 1:
        label = "多车事故"
        reason = "collision_evidence"
    elif eaca_version == "v3_evidence" and counts["close_multi_stop"] >= min_accident_clips and counts["close_multi_stop"] > counts["congestion"] and counts["close_multi_stop"] > counts["parking"]:
        label = "多车事故"
        reason = "repeated_close_multi_stop"
    elif eaca_version == "v2_soft" and counts["accident"] >= min_accident_clips and counts["accident"] >= counts["congestion"]:
        label = "多车事故"
        reason = "repeated_accident_evidence"
    elif counts["congestion"] >= math.ceil(n * 0.6):
        label = "拥堵"
        reason = "majority_congestion"
    elif counts["parking"] >= 1:
        label = "异常停车"
        reason = "local_parking_without_accident"
    else:
        label = label_counts.most_common(1)[0][0] if label_counts else None
        reason = "label_majority_fallback" if label else "no_evidence"

    return {
        "eaca_label": label,
        "eaca_reason": reason,
        "signal_counts": dict(counts),
        "clip_label_counts": dict(label_counts),
    }


def parse_plan(text):
    if not text:
        return DEFAULT_PLAN
    plan = {}
    for item in text.split(","):
        if not item.strip():
            continue
        label, n = item.split(":", 1)
        plan[label.strip()] = int(n)
    return plan


def main():
    ap = argparse.ArgumentParser(description="EACA smoke: event-aware clip aggregation for traffic accidents.")
    ap.add_argument("--meta", required=True)
    ap.add_argument("--test-jsonl", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--sampler-path", default="./holmesvau/ATS/anomaly_scorer.pth")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-clips", type=int, default=5)
    ap.add_argument("--frames-per-clip", type=int, default=6)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--plan", default="", help="Comma format: 多车事故:48,拥堵:60,异常停车:60")
    ap.add_argument("--min-accident-clips", type=int, default=2)
    ap.add_argument(
        "--eaca-version",
        choices=["v2_soft", "v3_evidence"],
        default="v2_soft",
        help="v2_soft restores the earlier high-recall soft fallback; v3_evidence is the stricter negative-control version.",
    )
    ap.add_argument("--output-jsonl", required=True)
    args = ap.parse_args()

    import torch
    from decord import VideoReader, cpu

    from holmesvau.holmesvau_utils import load_model

    rows = make_subset(read_jsonl(args.test_jsonl), parse_plan(args.plan), args.max_samples)
    video_root = load_video_root(args.meta)

    device = torch.device(args.device)
    model, tokenizer, generation_config, _ = load_model(args.model_path, args.sampler_path, device)
    generation_config = dict(generation_config)
    generation_config["max_new_tokens"] = 256

    outputs = []
    totals = Counter()
    pred_counts = Counter()
    correct = Counter()
    confusion = {label: Counter() for label in TARGET_LABELS}
    errors = Counter()

    for idx, row in enumerate(rows):
        video = row.get("video", "")
        gt_label = label_from_video(video)
        video_path = resolve_video_path(video_root, video)
        rec = {
            "index": idx,
            "id": row.get("id", idx),
            "video": video,
            "video_path": video_path,
            "gt_label": gt_label,
            "clips": [],
            "eaca_label": None,
            "eaca_reason": None,
            "error": None,
        }
        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            prompt = EACA_PROMPT_V2_SOFT if args.eaca_version == "v2_soft" else EACA_PROMPT_V3_EVIDENCE
            for clip_id in range(args.num_clips):
                frame_indices = clip_frame_indices(len(vr), clip_id, args.num_clips, args.frames_per_clip)
                raw = chat_clip(vr, frame_indices, prompt, model, tokenizer, generation_config)
                parsed = parse_clip_response(raw, eaca_version=args.eaca_version)
                rec["clips"].append({
                    "clip_id": clip_id,
                    "frame_indices": list(map(int, frame_indices)),
                    "raw": raw,
                    "parsed": parsed,
                })
            agg = aggregate_clips(rec["clips"], min_accident_clips=args.min_accident_clips, eaca_version=args.eaca_version)
            rec.update(agg)
            rec["eaca_version"] = args.eaca_version
        except Exception as exc:
            rec["error"] = str(exc)

        totals[gt_label] += 1
        if rec["error"]:
            errors[str(rec["error"])] += 1
        if rec["eaca_label"]:
            pred_counts[rec["eaca_label"]] += 1
            if gt_label in confusion:
                confusion[gt_label][rec["eaca_label"]] += 1
        if rec["eaca_label"] == gt_label:
            correct[gt_label] += 1
        outputs.append(rec)
        print(f"[{idx + 1}/{len(rows)}] gt={gt_label} pred={rec['eaca_label']} reason={rec['eaca_reason']} err={rec['error']}")

    write_jsonl(args.output_jsonl, outputs)

    print("\n=== EACA smoke summary ===")
    for label in TARGET_LABELS:
        total = totals[label]
        if not total:
            continue
        pred_total = pred_counts[label]
        precision = correct[label] / pred_total if pred_total else 0.0
        recall = correct[label] / total if total else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        print(label, {
            "total": total,
            "pred": pred_total,
            "correct_as_label": correct[label],
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })
    print("pred_counts:", dict(pred_counts))
    print("errors:", dict(errors))
    print("confusion:")
    for label in TARGET_LABELS:
        if totals[label]:
            print(label, dict(confusion[label]))
    print("output:", args.output_jsonl)


if __name__ == "__main__":
    main()
