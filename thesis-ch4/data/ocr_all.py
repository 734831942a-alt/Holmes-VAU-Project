"""Resumable OCR stage, file-locked so it may overlap independent timeline decoding."""
import argparse
import concurrent.futures
import fcntl
from pathlib import Path

from data.common import config, files, read_jsonl, append_jsonl, fingerprint, issue
from data.ocr import recognize


def run_ocr(paths, cfg):
    Path(cfg["ocr_lock"]).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg["ocr_lock"], "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = {r["video_id"]: r for r in read_jsonl(cfg["ocr_records"])}
        if any(r.get("ocr_config_sha256") != fingerprint(cfg["ocr"]) for r in rows.values()):
            raise RuntimeError("OCR cache configuration mismatch; archive obsolete cache explicitly")
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg["ocr"]["workers"]) as pool:
            futures = {pool.submit(recognize, p, cfg): p for p in paths if p.stem not in rows}
            for future in concurrent.futures.as_completed(futures):
                path = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    issue(cfg, "ocr-runtime", "OCR 引擎运行失败", str(path) + "\n" + str(exc))
                    raise
                append_jsonl(cfg["ocr_records"], row)
                rows[path.stem] = row
                if len(rows) % cfg["progress_every"] == 0:
                    print("ocr", len(rows), "/", len(paths), flush=True)
        return [rows[p.stem] for p in paths]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = config(args.config)
    rows = run_ocr(files(cfg), cfg)
    print("OCR complete", len(rows), flush=True)


if __name__ == "__main__":
    main()
