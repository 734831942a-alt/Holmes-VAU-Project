import argparse
import json
import os
from pathlib import Path

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    return "".join(str(text).strip().lower().split())


def resolve_video_path(video_root: str, video_item: str) -> str:
    p = Path(video_item)
    if p.is_absolute():
        return str(p)
    return str(Path(video_root) / video_item)


def extract_prompt_and_gt(sample: dict):
    # Prefer explicit fields; fallback to conversation format.
    prompt = str(sample.get("prompt", "")).strip()
    gt = str(sample.get("gt", "")).strip()
    convs = sample.get("conversations", [])
    for turn in convs:
        role = turn.get("from", "")
        value = str(turn.get("value", "")).strip()
        if role == "human" and not prompt:
            prompt = value
        if role == "gpt" and not gt:
            gt = value
    # Remove training placeholder, inference prompt should be plain question.
    prompt = prompt.replace("<video>", "").strip()
    return prompt, gt


def parse_args():
    parser = argparse.ArgumentParser(description="Batch test HolmesVAU on traffic_test.jsonl.")
    parser.add_argument("--meta", default="autodl-tmp/高架桥数据/traffic_meta.json", help="Path to traffic_meta.json")
    parser.add_argument(
        "--test-jsonl",
        default="autodl-tmp/高架桥数据/traffic_test.jsonl",
        help="Path to test jsonl",
    )
    parser.add_argument("--model-path", default="./ckpts/HolmesVAU-2B", help="Path to model directory")
    parser.add_argument(
        "--sampler-path", default="./holmesvau/ATS/anomaly_scorer.pth", help="Path to anomaly scorer checkpoint"
    )
    parser.add_argument("--device", default="cuda:0", help="Device, e.g. cuda:0 or cpu")
    parser.add_argument("--select-frames", type=int, default=12, help="Frames sampled for each video")
    parser.add_argument("--use-ats", action="store_true", help="Enable ATS sampling")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit number of test samples; 0 means all")
    parser.add_argument(
        "--output-jsonl",
        default="autodl-tmp/高架桥数据/traffic_test_pred.jsonl",
        help="Where to write per-sample predictions",
    )
    parser.add_argument(
        "--summary-json",
        default="autodl-tmp/高架桥数据/traffic_test_summary.json",
        help="Where to write summary",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    from holmesvau.holmesvau_utils import generate, load_model

    with open(args.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    ds_meta = next(iter(meta.values()))
    video_root = ds_meta["root"]

    with open(args.test_jsonl, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    device = torch.device(args.device)
    model, tokenizer, generation_config, sampler = load_model(args.model_path, args.sampler_path, device)

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ok = 0
    failed = 0
    exact_match = 0
    contain_match = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for idx, sample in enumerate(samples):
            sample_id = sample.get("id", idx)
            video_item = sample.get("video", "")
            prompt, gt = extract_prompt_and_gt(sample)
            video_path = resolve_video_path(video_root, video_item)

            rec = {
                "index": idx,
                "id": sample_id,
                "video": video_item,
                "video_path": video_path,
                "prompt": prompt,
                "gt": gt,
                "pred": None,
                "sampled_frames": [],
                "error": None,
            }

            try:
                pred, _, frame_indices, _ = generate(
                    video_path=video_path,
                    prompt=prompt,
                    model=model,
                    tokenizer=tokenizer,
                    generation_config=generation_config,
                    sampler=sampler,
                    select_frames=args.select_frames,
                    use_ATS=args.use_ats,
                )
                pred_text = "" if pred is None else str(pred).strip()
                if not pred_text:
                    raise ValueError("empty_response")
                rec["pred"] = pred_text
                rec["sampled_frames"] = list(map(int, frame_indices))

                gt_norm = normalize_text(gt)
                pred_norm = normalize_text(pred_text)
                if gt_norm and pred_norm and gt_norm == pred_norm:
                    exact_match += 1
                if gt_norm and pred_norm and (gt_norm in pred_norm or pred_norm in gt_norm):
                    contain_match += 1

                ok += 1
            except Exception as e:
                rec["error"] = str(e)
                failed += 1

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{idx + 1}/{len(samples)}] id={sample_id} ok={rec['error'] is None}")

    total = len(samples)
    summary = {
        "total": total,
        "ok": ok,
        "failed": failed,
        "exact_match_rate": (exact_match / ok) if ok else 0.0,
        "contain_match_rate": (contain_match / ok) if ok else 0.0,
        "meta": args.meta,
        "test_jsonl": args.test_jsonl,
        "model_path": args.model_path,
        "sampler_path": args.sampler_path,
        "select_frames": args.select_frames,
        "use_ats": args.use_ats,
        "output_jsonl": str(out_path),
    }

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print("Summary:", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
