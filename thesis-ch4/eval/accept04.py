"""Executable one-to-one SPEC-04 §6 checklist; missing/blocked results never pass."""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import subprocess

from data.common import config, files, read_jsonl, write_json, fingerprint
from data.timeline import normalize_pts
from data.make_splits import leakage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    ac = config(args.config)
    dc, bc = config(ac["data_config"]), config(ac["b0_config"])
    checks = []

    def check(identifier, criterion, function, applicable=True):
        try:
            passed, evidence = function()
        except Exception as exc:
            passed, evidence = False, {"error": type(exc).__name__ + ": " + str(exc)}
        checks.append({"id": identifier, "criterion": criterion, "applicable": applicable,
                       "passed": bool(passed), "evidence": evidence})

    manifest = read_jsonl(dc["manifest"])
    paths = files(dc)
    proof = {r["video_id"]: r for r in read_jsonl(dc["timeline_records"])}
    b0 = Path(ac["b0_output"])

    def manifest_check():
        ids = [r["video_id"] for r in manifest]
        valid = len(ids) == len(paths) and set(ids) == {p.stem for p in paths}
        valid = valid and all(set(ac["required_manifest_fields"]) <= set(r) and all(r[k] is not None for k in ac["required_manifest_fields"]) for r in manifest)
        return valid, {"manifest_rows": len(manifest), "self_video_count": len(paths)}

    check("6.1", "manifest 行数等于自采视频数且所有字段完整", manifest_check)

    def report_check():
        text = Path(dc["timeline_report"]).read_text()
        sample = read_jsonl(dc["probe_records"])
        valid = [r for r in sample if "error" not in r]
        example_ids = [r["video_id"] for r in proof.values() if "error" not in r][:dc["example_count"]]
        present = sum(r["video_id"] in text for r in proof.values() if "error" not in r)
        return (len(valid) == dc["probe_samples"] and "nonzero_start" in text and "dur_mismatch" in text and "ratios" in text and present >= dc["example_count"]), {"sample_count": len(sample), "valid_sample_count": len(valid), "real_examples_in_report": present}

    check("6.2", "timeline_report 含两类异常比例和三个真实样例", report_check)

    def timeline_check():
        failures = []
        for row in manifest:
            p = proof[row["video_id"]]
            times = normalize_pts(p["pts"], p["time_base"])
            expected_flags = {flag for flag in p["flags"]}
            if (times[0] != 0 or min(times) != 0 or row["duration_sec"] != float(times[-1]) or
                row["duration_sec"] <= 0 or row["first_frame_pts"] != p["pts"][0] or
                row["time_base"] != p["time_base"] or not expected_flags <= set(row["flags"])):
                failures.append(row["video_id"])
        return len(manifest) == len(paths) and not failures, {"checked": len(manifest), "failures": failures}

    check("6.3", "逐视频首帧/最小时间为零，duration 是解码末帧且大于零", timeline_check)

    def camera_check():
        groups = json.loads(Path(dc["camera_groups"]).read_text())
        uncertain = groups.pop("uncertain")
        coverage = Counter(vid for values in groups.values() for vid in values)
        expected = Counter(r["video_id"] for r in manifest)
        uncertain_ids = {r["video_id"] for r in uncertain}
        unknown_ids = {r["video_id"] for r in manifest if r["camera_group"].startswith("UNKNOWN_")}
        reviewable = all({"raw_ocr", "normalized", "confidence", "roi", "frame_index", "reason"} <= set(r) for r in uncertain)
        matching = all(r["video_id"] in groups[r["camera_group"]] for r in manifest)
        return (coverage == expected and len(coverage) == len(paths) and uncertain_ids == unknown_ids and reviewable and matching), {"group_count": len(groups), "covered": len(coverage), "uncertain_count": len(uncertain), "reviewable": reviewable}

    check("6.4", "camera_groups 全覆盖且 uncertain 单列可复核", camera_check)

    def split_check():
        splits = json.loads(Path(dc["splits"]).read_text())
        result = leakage(manifest, splits, dc)
        lock = json.loads(Path(dc["split_lock"]).read_text())
        membership = {v: s for s, ids in splits.items() for v in ids}
        result["test_lock_matches"] = sorted(lock["test"]) == sorted(splits["test"])
        result["manifest_split_matches"] = all(r["split"] == membership[r["video_id"]] for r in manifest)
        return (result["cross_split_leak"] == 0 and result["source_event_cross_split_leak"] == 0 and result["test_lock_matches"] and result["manifest_split_matches"]), result

    check("6.5", "cross_split_leak == 0；源事件隔离且 test 锁定", split_check)

    def table_check():
        table = Path(dc["split_table"]).read_text()
        counts = {c: {s: sum(r["category"] == c and r["split"] == s for r in manifest) for s in dc["split_ratios"]} for c in dc["baseline_categories"]}
        zero = [(c, s) for c in dc["rare_categories"] for s in dc["split_ratios"] if counts[c][s] == 0]
        rare_issue = Path(ac["rare_class_issue"])
        matching = all("| " + c + " | " + " | ".join(str(v) for v in counts[c].values()) + " |" in table for c in counts)
        return matching and (not zero or rare_issue.exists()), {"counts": counts, "rare_zeros": zero, "rare_issue_exists": rare_issue.exists()}

    check("6.6", "六类 × 三 split 表和稀有类零数量 issue", table_check)

    def metrics_check():
        value = json.loads((b0 / bc["metrics_file"]).read_text())
        from eval.b0_selfdata import metrics
        selected = [r for r in manifest if r["split"] == "test"]
        recalculated = metrics(read_jsonl(b0 / bc["predictions_file"]), selected, bc)
        return set(ac["required_metrics"]) <= set(value) and value == recalculated and value["complete"], value

    check("6.7", "B0 含六类规定指标，重算一致且无正式 IoU", metrics_check)

    def predictions_check():
        rows = read_jsonl(b0 / bc["predictions_file"])
        test = json.loads(Path(dc["splits"]).read_text())["test"]
        ids = [r["video_id"] for r in rows]
        from eval.b0_selfdata import prediction
        valid = len(ids) == len(test) and set(ids) == set(test) and all(set(ac["required_prediction_fields"]) <= set(r) for r in rows)
        valid = valid and all(prediction(r["video_id"], r["raw_output"], r["frames_seen"], bc) == r for r in rows)
        attempts = read_jsonl(b0 / bc["attempts_file"])
        valid = valid and Counter(r["video_id"] for r in attempts) == Counter(ids)
        attempted_frames = {r["video_id"]: r["frame_indices"] for r in attempts}
        valid = valid and all(r["frames_seen"] == len(attempted_frames[r["video_id"]]) for r in rows)
        run = json.loads((b0 / bc["run_file"]).read_text())
        unchanged = (run["config_sha256"] == fingerprint(bc) and run["data_config_sha256"] == fingerprint(dc) and
                     run["manifest_sha256"] == fingerprint(manifest) and run["prompt_template"] == bc["prompt_template"])
        return valid and unchanged, {"prediction_rows": len(rows), "test_rows": len(test), "failed_rows_retained": sum(not r["parse_ok"] for r in rows), "attempts": len(attempts), "run_configuration_unchanged": unchanged}

    check("6.8", "predictions 每个 test 样本恰好一行，失败保留且不重试", predictions_check)
    if bc["pilot_enabled"]:
        check("6.9", "人工 pilot30 含粗 mIoU 和起点误差中位数", lambda: (set(ac["required_pilot_fields"]) <= set(json.loads((b0 / bc["pilot_file"]).read_text())), str(b0 / bc["pilot_file"])))
    else:
        check("6.9", "可选人工 pilot30", lambda: (True, "未执行：无人工粗标注；不生成模型或脚本伪标注"), applicable=False)

    def third_check():
        p = subprocess.run(["git", "-C", dc["third_chapter"], "status", "--porcelain"], text=True, capture_output=True)
        return p.returncode == 0 and not p.stdout.strip(), {"exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr, "issue": ac["third_chapter_issue"]}

    check("6.10", "第三章 git status --porcelain 为空", third_check)

    def frozen_check():
        before = json.loads(Path(ac["frozen_hashes"]).read_text())
        changed = [name for name, digest in before.items() if not Path(name).exists() or hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest]
        return not changed, {"checked": len(before), "changed": changed}

    extras = []
    try:
        frozen, frozen_evidence = frozen_check()
    except Exception as exc:
        frozen, frozen_evidence = False, str(exc)
    packages = {name: version(name) for name in ac["protected_packages"]}
    try:
        forbidden = version(ac["forbidden_package"])
    except PackageNotFoundError:
        forbidden = None
    tests = json.loads(Path(dc["test_results"]).read_text()) if Path(dc["test_results"]).exists() else {"successful": False}
    contract_hash = hashlib.sha256(Path(ac["frozen_spec"]).read_bytes()).hexdigest()
    all_pass = all(c["passed"] for c in checks) and frozen and tests["successful"] and packages == ac["protected_packages"] and forbidden is None and contract_hash == ac["frozen_spec_sha256"]
    result = {"spec": "SPEC-04", "status": "passed" if all_pass else "blocked", "checks": checks,
              "summary": {"passed": sum(c["passed"] and c["applicable"] for c in checks),
                          "failed": sum(not c["passed"] and c["applicable"] for c in checks),
                          "skipped": sum(not c["applicable"] for c in checks), "total": len(checks)},
              "frozen_assets": {"passed": frozen, "evidence": frozen_evidence}, "unit_tests": tests,
              "protected_packages": packages, "protected_packages_unchanged": packages == ac["protected_packages"],
              "flash_attn": forbidden, "contract_sha256": contract_hash, "contract_hash_matches": contract_hash == ac["frozen_spec_sha256"],
              "new_system_packages": ac["new_system_packages"],
              "issues": sorted(str(p) for p in Path(dc["issues_dir"]).glob(ac["issue_glob"]))}
    write_json(dc["acceptance"], result)
    print(json.dumps(result["summary"]))
    raise SystemExit(not all_pass)


if __name__ == "__main__":
    main()
