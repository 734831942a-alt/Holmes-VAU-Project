"""Fixed-ROI OCR only; no annotation text or filenames used as camera identifiers."""
import csv
import io
import os
import re
import subprocess
import unicodedata
import threading
from pathlib import Path

from data.common import fingerprint

_local = threading.local()


def recognize(path, cfg, save=None):
    import cv2
    o = cfg["ocr"]
    if o["engine"] not in {"tesseract", "rapidocr"}:
        raise ValueError("Unsupported configured OCR engine")
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, o["frame_index"])
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError("Cannot decode OCR frame: " + str(path))
    height, width = frame.shape[:2]
    left, top, right, bottom = o["roi"]
    crop = frame[round(top * height):round(bottom * height), round(left * width):round(right * width)]
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save), crop)
    if o["engine"] == "rapidocr":
        from rapidocr_onnxruntime import RapidOCR
        key = fingerprint(o["rapid_options"])
        if getattr(_local, "key", None) != key:
            _local.engine = RapidOCR(**o["rapid_options"])
            _local.key = key
        found, _ = _local.engine(crop)
        found = found or []
        raw = " ".join(item[1] for item in found)
        confidence = min((float(item[2]) for item in found), default=o["confidence_floor"])
        normalized = "".join(unicodedata.normalize(o["unicode_form"], raw).split())
        valid = (len(found) == o["expected_text_lines"] and len(normalized) >= o["min_chars"] and
                 bool(re.search(o["content_regex"], normalized)))
        return {"video_id": path.stem, "raw_ocr": raw, "normalized": normalized,
                "confidence": confidence, "eligible": bool(valid and confidence >= o["confidence_threshold"]),
                "roi": o["roi"], "frame_index": o["frame_index"], "crop": str(save) if save else None,
                "engine": o["engine"], "ocr_config_sha256": fingerprint(o), "text_boxes": found}
    ok, encoded = cv2.imencode(".png", crop)
    if not ok:
        raise ValueError("Cannot encode OCR crop")
    env = dict(os.environ, OMP_THREAD_LIMIT=o["omp_threads"])
    result = subprocess.run([o["executable"], "stdin", "stdout", "-l", o["language"],
                             "--psm", str(o["psm"]), "tsv"], input=encoded.tobytes(),
                            capture_output=True, env=env, timeout=o["timeout_sec"])
    if result.returncode:
        raise ValueError(result.stderr.decode(errors="replace"))
    tokens = [r for r in csv.DictReader(io.StringIO(result.stdout.decode()), delimiter="\t")
              if r["text"].strip() and float(r["conf"]) >= o["confidence_floor"]]
    raw = " ".join(r["text"] for r in tokens)
    normalized = "".join(unicodedata.normalize(o["unicode_form"], raw).split())
    confidence = min((float(r["conf"]) / o["confidence_scale"] for r in tokens), default=o["confidence_floor"])
    valid = len(normalized) >= o["min_chars"] and bool(re.search(o["content_regex"], normalized))
    return {"video_id": path.stem, "raw_ocr": raw, "normalized": normalized,
            "confidence": confidence, "eligible": bool(valid and confidence >= o["confidence_threshold"]),
            "roi": o["roi"], "frame_index": o["frame_index"], "crop": str(save) if save else None,
            "engine": o["engine"], "ocr_config_sha256": fingerprint(o)}
