"""SPEC-04 manifest construction, including every decoded video and OCR uncertainty."""
import argparse
import concurrent.futures
import json
from pathlib import Path

from data.common import config, files, read_jsonl, write_json, write_jsonl, append_jsonl, issue, fingerprint
from data.probe_selfdata import scan, timeline_report
from data.ocr import recognize
from data.cameras import group_cameras
from data.ocr_all import run_ocr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = config(args.config)
    paths = files(cfg, args.src)
    # The prerequisite report must exist before the full manifest step.
    if not Path(cfg["probe_report"]).exists() or not Path(cfg["timeline_report"]).exists():
        raise RuntimeError("Run the §3 prerequisite probe first")
    if not Path(cfg["timeline_records"]).exists():
        write_jsonl(cfg["timeline_records"], read_jsonl(cfg["probe_records"]))
    rows = scan(paths, cfg, cfg["timeline_records"], True)
    timeline_report(rows, cfg, "全量解码视频")
    errors = [r for r in rows if "error" in r]
    if errors:
        issue(cfg, "timeline-decode", "全量解码存在无法建表的视频", json.dumps(errors, ensure_ascii=False, indent=2) + "\n停止 manifest/划分/B0，不跳过失败视频。")
        raise RuntimeError("Unresolved timeline failures; see issue")
    ocr_rows = run_ocr(paths, cfg)
    assignments, groups = group_cameras(ocr_rows, cfg)
    manifest = []
    for row in rows:
        group, confidence = assignments[row["video_id"]]
        manifest.append({"video_id": row["video_id"], "path": row["path"],
                         "category": row["video_id"].split(cfg["category_separator"])[0],
                         "duration_sec": row["duration_sec"], "fps_used": row["fps_used"],
                         "first_frame_pts": row["first_frame_pts"], "time_base": row["time_base"],
                         "camera_group": group, "camera_group_conf": confidence,
                         "split": None, "flags": row["flags"]})
    write_json(cfg["camera_groups"], groups)
    write_jsonl(args.out, manifest)
    print("manifest complete", len(manifest), "uncertain", len(groups["uncertain"]), flush=True)


if __name__ == "__main__":
    main()
