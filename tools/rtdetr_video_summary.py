import json
from pathlib import Path
import cv2
from ultralytics import RTDETR

KEEP = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

def sample_indices(n, k=12):
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    step = n / k
    return [int(i * step) for i in range(k)]

def summarize_video(video_path, model, conf=0.25, k=12):
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = set(sample_indices(n, k))
    cnt = {v: 0 for v in KEEP.values()}
    max_cnt = {v: 0 for v in KEEP.values()}

    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi in idxs:
            r = model.predict(frame, conf=conf, verbose=False)[0]
            frame_cnt = {v: 0 for v in KEEP.values()}
            if r.boxes is not None and len(r.boxes) > 0:
                classes = r.boxes.cls.int().tolist()
                for c in classes:
                    if c in KEEP:
                        frame_cnt[KEEP[c]] += 1
            for k2, v in frame_cnt.items():
                cnt[k2] += v
                if v > max_cnt[k2]:
                    max_cnt[k2] = v
        fi += 1

    cap.release()
    return {"mean_count": cnt, "max_count": max_cnt, "sampled_frames": len(idxs)}

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="/root/autodl-tmp/models/rtdetr-l.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--frames", type=int, default=12)
    args = ap.parse_args()

    model = RTDETR(args.weights)
    root = Path(args.video_root)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    with open(args.jsonl, "r", encoding="utf-8") as f, open(outp, "w", encoding="utf-8") as w:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            vp = root / o["video"]
            rec = {"video": o["video"], "det": None, "ok": False}
            if vp.exists():
                try:
                    rec["det"] = summarize_video(vp, model, conf=args.conf, k=args.frames)
                    rec["ok"] = True
                except Exception as e:
                    rec["err"] = str(e)
            else:
                rec["err"] = "video_not_found"
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("saved:", outp)

if __name__ == "__main__":
    main()
