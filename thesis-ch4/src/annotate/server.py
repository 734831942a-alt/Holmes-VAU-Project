"""Minimal human-only temporal annotation server for SPEC-02."""

import argparse
import fcntl
import json
import mimetypes
import os
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict
from urllib.parse import parse_qs, urlparse


class AnnotationStore:
    def __init__(self, manifest_path: Path, output_path: Path):
        self.rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.by_id = {row["video_id"]: row for row in self.rows}
        if len(self.by_id) != len(self.rows):
            raise ValueError("manifest contains duplicate video_id values")
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.touch(exist_ok=True)
        self.sessions: Dict[str, Dict] = {}
        self.lock = threading.Lock()

    def completed(self) -> Dict[str, Dict]:
        records = {}
        with self.output_path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    records[record["video_id"]] = record
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return records

    def next_item(self):
        done = self.completed()
        for row in self.rows:
            if row["video_id"] not in done:
                token = uuid.uuid4().hex
                with self.lock:
                    self.sessions[token] = {
                        "video_id": row["video_id"],
                        "started": time.monotonic(),
                    }
                public = {key: value for key, value in row.items() if key != "path"}
                public["session_id"] = token
                public["progress"] = {"completed": len(done), "total": len(self.rows)}
                return public
        return {"done": True, "progress": {"completed": len(done), "total": len(self.rows)}}

    def submit(self, payload: Dict) -> Dict:
        required = {
            "video_id", "t_start", "t_end", "start_observable",
            "spans_whole_clip", "annotator", "session_id",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError("missing fields: " + ", ".join(missing))
        video_id = str(payload["video_id"])
        row = self.by_id.get(video_id)
        if row is None:
            raise ValueError("video_id is not in manifest")
        if not str(payload["annotator"]).strip():
            raise ValueError("annotator is required")
        if type(payload["start_observable"]) is not bool:
            raise ValueError("start_observable must be boolean")
        if type(payload["spans_whole_clip"]) is not bool:
            raise ValueError("spans_whole_clip must be boolean")
        try:
            start = float(payload["t_start"])
            end = float(payload["t_end"])
        except (TypeError, ValueError) as exc:
            raise ValueError("t_start and t_end must be numeric") from exc
        duration = float(row["duration"])
        if not (0 <= start < end <= duration):
            raise ValueError("times must satisfy 0 <= t_start < t_end <= duration")

        token = str(payload["session_id"])
        with self.lock:
            session = self.sessions.pop(token, None)
        if session is None or session["video_id"] != video_id:
            raise ValueError("annotation session is missing, expired, or mismatched")
        elapsed = round(max(0.0, time.monotonic() - session["started"]), 3)
        record = {
            "video_id": video_id,
            "category": row["category"],
            "duration": duration,
            "t_start": start,
            "t_end": end,
            "start_observable": payload["start_observable"],
            "spans_whole_clip": payload["spans_whole_clip"],
            "notes": str(payload.get("notes", "")),
            "annotation_seconds": elapsed,
            "annotator": str(payload["annotator"]).strip(),
        }

        with self.output_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            existing = {
                json.loads(line)["video_id"]
                for line in handle
                if line.strip()
            }
            if video_id in existing:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                raise FileExistsError("video already annotated")
            handle.seek(0, os.SEEK_END)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return record


def make_handler(store: AnnotationStore, html_path: Path):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, value, status=HTTPStatus.OK):
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = html_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/next":
                self.send_json(store.next_item())
                return
            if parsed.path == "/video":
                video_id = parse_qs(parsed.query).get("id", [""])[0]
                row = store.by_id.get(video_id)
                if row is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self.send_video(Path(row["path"]))
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def send_video(self, path: Path):
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            size = path.stat().st_size
            start, end = 0, size - 1
            range_header = self.headers.get("Range")
            status = HTTPStatus.OK
            if range_header:
                try:
                    unit, value = range_header.split("=", 1)
                    if unit != "bytes":
                        raise ValueError
                    first, last = value.split("-", 1)
                    start = int(first) if first else 0
                    end = int(last) if last else size - 1
                    end = min(end, size - 1)
                    if start < 0 or start > end:
                        raise ValueError
                    status = HTTPStatus.PARTIAL_CONTENT
                except ValueError:
                    self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    return
            length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def do_POST(self):
            if urlparse(self.path).path != "/api/annotation":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                record = store.submit(payload)
                self.send_json(record, HTTPStatus.CREATED)
            except FileExistsError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.CONFLICT)
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, fmt, *args):
            print(f"{self.address_string()} - {fmt % args}", flush=True)

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()
    store = AnnotationStore(Path(args.manifest), Path(args.output))
    html_path = Path(__file__).with_name("index.html")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store, html_path))
    print(f"annotation server: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
