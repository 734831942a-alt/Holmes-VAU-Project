"""Decode presentation timestamps: first frame zero, last decoded frame as duration."""
import json
import subprocess
from fractions import Fraction


def normalize_pts(pts, time_base):
    if not pts or any(p is None for p in pts):
        raise ValueError("Missing decoded frame PTS")
    first = pts[0]
    values = [(p - first) * Fraction(time_base) for p in pts]
    if min(values) != 0 or values[-1] <= 0:
        raise ValueError("First-frame-zero/minimum-zero or positive-duration invariant failed")
    if any(b < a for a, b in zip(values, values[1:])):
        raise ValueError("Decoded presentation timeline is non-monotonic")
    return values


def probe(path, cfg, decode=True):
    command = [cfg["ffprobe"], "-v", "error", "-threads", str(cfg["decode_threads"]),
               "-select_streams", cfg["stream_selector"], "-show_streams", "-show_format"]
    if decode:
        command += ["-show_frames", "-show_entries",
                    "frame=" + cfg["frame_pts_field"] + ":stream=index,start_time,nb_frames,avg_frame_rate,time_base,width,height,duration:format=duration"]
    command += ["-of", "json", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=cfg["timeout_sec"])
    if result.returncode:
        raise ValueError("ffprobe decoding error: " + result.stderr)
    data = json.loads(result.stdout)
    if len(data["streams"]) != 1:
        raise ValueError("Expected one video stream; multiple/absent streams require a contract decision")
    stream = data["streams"][cfg["stream_index"]]
    row = {"video_id": path.stem, "path": str(path), "stream": stream,
           "container_duration": data["format"].get("duration"), "decoder_diagnostics": result.stderr.strip()}
    if not decode:
        return row
    pts = [frame.get(cfg["frame_pts_field"]) for frame in data["frames"]]
    normalized = normalize_pts(pts, stream["time_base"])
    if not stream.get("start_time") or not stream.get("avg_frame_rate"):
        raise ValueError("Missing required stream timeline metadata")
    nonzero = Fraction(stream["start_time"]) != 0 or pts[0] != 0
    mismatch = row["container_duration"] is None or Fraction(row["container_duration"]) != normalized[-1]
    row.update(first_frame_pts=pts[0], last_frame_pts=pts[-1], pts=pts,
               time_base=stream["time_base"], duration_sec=float(normalized[-1]),
               min_time_sec=float(min(normalized)), fps_used=float(Fraction(stream["avg_frame_rate"])),
               flags=(["nonzero_start"] if nonzero else []) + (["dur_mismatch"] if mismatch else []) +
               (["decode_warning"] if row["decoder_diagnostics"] else []))
    return row
