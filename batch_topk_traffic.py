import argparse
import json
import re
from pathlib import Path


DEFAULT_LABELS = ["多车事故", "拥堵", "异常停车", "占道施工", "二轮车辆闯入", "抛洒物"]


def parse_csv(text):
    return [x.strip() for x in (text or "").split(",") if x.strip()]


def resolve_video_path(video_root, video_item):
    p = Path(video_item)
    if p.is_absolute():
        return str(p)
    return str(Path(video_root) / video_item)


def extract_prompt_and_gt(sample):
    prompt = str(sample.get("prompt", "")).strip()
    gt = str(sample.get("gt", "")).strip()
    for turn in sample.get("conversations", []):
        role = turn.get("from", "")
        value = str(turn.get("value", "")).strip()
        if role == "human" and not prompt:
            prompt = value
        if role == "gpt" and not gt:
            gt = value
    return prompt.replace("<video>", "").strip(), gt


def load_jsonl_map(path):
    data = {}
    if not path:
        return data
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            video = obj.get("video", "")
            if video:
                data[video] = obj
    return data


def detect_label(text, labels):
    text = "" if text is None else str(text)
    hits = []
    for label in labels:
        pos = text.find(label)
        if pos >= 0:
            hits.append((pos, label))
    hits.sort(key=lambda x: x[0])
    out = []
    for _, label in hits:
        if label not in out:
            out.append(label)
    return out


def make_topk_prompt(base_prompt, labels, top_k):
    label_text = "、".join(labels)
    return (
        "请只根据视频内容完成交通事件类别排序。\n"
        f"候选类别只能从以下六类中选择：{label_text}。\n"
        f"请按可能性从高到低输出最可能的{top_k}类。\n"
        f"严格输出格式：TOP{top_k}: 类别1, 类别2, 类别3\n"
        "不要输出解释，不要输出候选类别之外的文字。\n"
        f"原始问题：{base_prompt}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate top-k traffic labels for rerank/fusion experiments.")
    parser.add_argument("--meta", required=True)
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--base-pred-jsonl", default="", help="Optional baseline prediction jsonl; copied into pred field.")
    parser.add_argument("--base-pred-field", default="pred")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--sampler-path", default="./holmesvau/ATS/anomaly_scorer.pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--select-frames", type=int, default=12)
    parser.add_argument("--use-ats", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-jsonl", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    from holmesvau.holmesvau_utils import generate, load_model

    labels = parse_csv(args.labels)
    with open(args.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    video_root = next(iter(meta.values()))["root"]

    with open(args.test_jsonl, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    base_pred_map = load_jsonl_map(args.base_pred_jsonl)

    device = torch.device(args.device)
    model, tokenizer, generation_config, sampler = load_model(args.model_path, args.sampler_path, device)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ok = 0
    failed = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for idx, sample in enumerate(samples):
            sample_id = sample.get("id", idx)
            video_item = sample.get("video", "")
            base_prompt, gt = extract_prompt_and_gt(sample)
            video_path = resolve_video_path(video_root, video_item)
            base_pred_rec = base_pred_map.get(video_item, {})
            base_pred = base_pred_rec.get(args.base_pred_field)

            rec = {
                "index": idx,
                "id": sample_id,
                "video": video_item,
                "video_path": video_path,
                "gt": gt,
                "pred": base_pred,
                "topk_raw": None,
                "topk_labels": [],
                "sampled_frames": [],
                "error": None,
            }

            try:
                topk_prompt = make_topk_prompt(base_prompt, labels, args.top_k)
                response, _, frame_indices, _ = generate(
                    video_path=video_path,
                    prompt=topk_prompt,
                    model=model,
                    tokenizer=tokenizer,
                    generation_config=generation_config,
                    sampler=sampler,
                    select_frames=args.select_frames,
                    use_ATS=args.use_ats,
                )
                raw = "" if response is None else str(response).strip()
                topk_labels = detect_label(raw, labels)[: args.top_k]
                rec["topk_raw"] = raw
                rec["topk_labels"] = topk_labels
                rec["sampled_frames"] = list(map(int, frame_indices))
                ok += 1
            except Exception as exc:
                rec["error"] = str(exc)
                failed += 1

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{idx + 1}/{len(samples)}] id={sample_id} ok={rec['error'] is None} topk={rec['topk_labels']}")

    print("Done.")
    print(json.dumps({
        "total": len(samples),
        "ok": ok,
        "failed": failed,
        "output_jsonl": str(out_path),
        "use_ats": args.use_ats,
        "select_frames": args.select_frames,
        "top_k": args.top_k,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
