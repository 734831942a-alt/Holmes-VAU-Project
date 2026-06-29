import json
from pathlib import Path
import cv2
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

TEXT_QUERIES = [
    "car", "truck", "bus", "motorcycle", "person",
    "traffic cone", "construction vehicle", "road barrier",
    "debris", "road obstacle", "scattered object"
]

def sample_indices(n, k=12):
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    step = n / k
    return [int(i * step) for i in range(k)]

def summarize_video(video_path, processor, model, device, box_threshold=0.30, text_threshold=0.25, frames=12):
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = set(sample_indices(n, frames))

    labels_all = {q: 0 for q in TEXT_QUERIES}
    labels_peak = {q: 0 for q in TEXT_QUERIES}
    used = 0
    fi = 0
    bad_reads = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            bad_reads += 1
            if bad_reads > 5:
                break
            fi += 1
            continue
        bad_reads = 0

        if fi in idxs:
            used += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = processor(images=rgb, text=TEXT_QUERIES, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            target_sizes = torch.tensor([rgb.shape[:2]], device=device)
            results = processor.post_process_grounded_object_detection(
                outputs=outputs,
                input_ids=inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=target_sizes
            )[0]

            frame_cnt = {q: 0 for q in TEXT_QUERIES}
            for lb in results["labels"]:
                label = str(lb)
                if label in frame_cnt:
                    frame_cnt[label] += 1

            for k, v in frame_cnt.items():
                labels_all[k] += v
                if v > labels_peak[k]:
                    labels_peak[k] = v

        fi += 1

    cap.release()
    return {"sum_count": labels_all, "peak_count": labels_peak, "sampled_frames": used}

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-id", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--box-thres", type=float, default=0.30)
    ap.add_argument("--text-thres", type=float, default=0.25)
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id).to(device).eval()

    root = Path(args.video_root)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(args.jsonl, "r", encoding="utf-8") as f, open(outp, "w", encoding="utf-8") as w:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            video = obj.get("video", "")
            vp = root / video
            rec = {"video": video, "ok": False, "ovd": None}
            if vp.exists():
                try:
                    rec["ovd"] = summarize_video(
                        vp, processor, model, device,
                        box_threshold=args.box_thres,
                        text_threshold=args.text_thres,
                        frames=args.frames
                    )
                    rec["ok"] = True
                except Exception as e:
                    rec["err"] = str(e)
            else:
                rec["err"] = "video_not_found"

            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += 1
            if args.max_samples > 0 and total >= args.max_samples:
                break

    print("saved:", outp, "records:", total)

if __name__ == "__main__":
    main()
