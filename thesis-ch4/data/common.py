"""Shared configuration and durable output helpers for SPEC-04."""
import hashlib
import json
import os
from pathlib import Path

import yaml


def config(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_text(path, text):
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(dest.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(dest)


def write_jsonl(path, rows):
    write_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def append_jsonl(path, row):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def issue(cfg, slug, title, detail):
    path = Path(cfg["issues_dir"]) / cfg["issue_template"].format(slug=slug)
    write_text(path, "# SPEC-04 · " + title + "\n\n" + detail + "\n")
    return str(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def files(cfg, src=None):
    from src.evidence.queries import LABELS
    paths = sorted(Path(src or cfg["src"]).glob(cfg["pattern"]))
    for path in paths:
        if path.stem.split(cfg["category_separator"])[0] not in LABELS:
            issue(cfg, "unknown-category", "无法映射到冻结六类", str(path))
            raise ValueError("Unknown category: " + str(path))
    return paths
