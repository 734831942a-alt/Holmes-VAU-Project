"""Behavior tests: leak rejection, real timeline arithmetic, durable failed predictions."""
import argparse
import copy
from pathlib import Path
import tempfile
import sys
import types
import unittest
from unittest.mock import Mock, patch
import yaml
import numpy as np

from data.common import config, append_jsonl, read_jsonl, write_json, write_jsonl
from data.timeline import normalize_pts
from data.cameras import group_cameras
from data.make_splits import assign, leakage
from eval.b0_selfdata import prediction

FIXTURE = None


class Spec04Tests(unittest.TestCase):
    def test_cross_split_leak_zero_and_source_events_indivisible(self):
        rows = FIXTURE["fixture_rows"]
        splits, checks = assign(rows, config(FIXTURE["data_config"]))
        self.assertEqual(checks["cross_split_leak"], 0)
        self.assertEqual(checks["source_event_cross_split_leak"], 0)
        where = {vid: name for name, vids in splits.items() for vid in vids}
        self.assertEqual(where[rows[0]["video_id"]], where[rows[1]["video_id"]])
        self.assertEqual(where[rows[0]["video_id"]], where[rows[2]["video_id"]])
        bad = copy.deepcopy(splits)
        vid = rows[0]["video_id"]
        old = where[vid]
        new = next(n for n in splits if n != old)
        bad[old].remove(vid)
        bad[new].append(vid)
        self.assertGreater(leakage(rows, bad, config(FIXTURE["data_config"]))["cross_split_leak"], 0)

    def test_test_membership_locked_across_order_and_seed_change(self):
        cfg = config(FIXTURE["data_config"])
        rows = FIXTURE["fixture_rows"]
        splits, _ = assign(rows, cfg)
        cfg["seed"] += 1
        repeated, _ = assign(list(reversed(rows)), cfg, splits["test"])
        self.assertEqual(splits["test"], repeated["test"])
        incomplete = [rows[0]["video_id"]]
        with self.assertRaises(ValueError):
            assign(rows, cfg, incomplete)

    def test_first_frame_zero_and_last_decoded_pts(self):
        for case in FIXTURE["timeline_cases"]:
            actual = normalize_pts(case["pts"], case["time_base"])
            self.assertEqual([float(v) for v in actual], case["expected"])
            self.assertEqual(min(actual), 0)
            self.assertEqual(actual[0], 0)
            self.assertGreater(actual[-1], 0)
        for pts in FIXTURE["bad_timelines"]:
            with self.assertRaises(ValueError):
                normalize_pts(pts, FIXTURE["test_time_base"])

    def test_parse_failures_are_persisted_with_raw_text(self):
        cfg = config(FIXTURE["b0_config"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / FIXTURE["fixture_prediction_path"]
            for output in FIXTURE["invalid_outputs"]:
                record = prediction(FIXTURE["fixture_video_id"], output, FIXTURE["frames_seen"], cfg)
                self.assertFalse(record["parse_ok"])
                append_jsonl(path, record)
            rows = read_jsonl(path)
            self.assertEqual(len(rows), len(FIXTURE["invalid_outputs"]))
            self.assertEqual([r["raw_output"] for r in rows], FIXTURE["invalid_outputs"])
            self.assertTrue(all(r["failure_reason"] == "parse_failure" for r in rows))

    def test_parser_preserves_unbounded_output_without_clipping(self):
        row = prediction(FIXTURE["fixture_video_id"], FIXTURE["valid_output"], FIXTURE["frames_seen"], config(FIXTURE["b0_config"]))
        self.assertTrue(row["parse_ok"])
        self.assertEqual(row["parsed_start"], FIXTURE["valid_start"])
        self.assertEqual(row["parsed_end"], FIXTURE["valid_end"])

    def test_ambiguous_ocr_chain_and_low_confidence_stay_unknown(self):
        cfg = config(FIXTURE["data_config"])
        assignments, groups = group_cameras(FIXTURE["ocr_fixtures"], cfg)
        self.assertEqual(len(groups["uncertain"]), len(FIXTURE["ocr_fixtures"]))
        self.assertTrue(all(value[0].startswith("UNKNOWN_") for value in assignments.values()))

    def test_new_source_root_cannot_reuse_old_timeline_cache(self):
        from data.probe_selfdata import scan
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = config(FIXTURE["data_config"])
            cfg["issues_dir"] = str(root)
            old = root / FIXTURE["old_source_dir"] / FIXTURE["integration_source_file"]
            new = root / FIXTURE["new_source_dir"] / FIXTURE["integration_source_file"]
            cache = root / FIXTURE["cache_file"]
            write_jsonl(cache, [{"video_id": old.stem, "path": str(old)}])
            with self.assertRaisesRegex(ValueError, "source path changed"):
                scan([new], cfg, cache, True)

    def test_b0_resume_does_not_retry_failed_parse_or_drop_record(self):
        from eval.b0_selfdata import main as b0_main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dc = config(FIXTURE["data_config"])
            bc = config(FIXTURE["b0_config"])
            for field, fixture in [("splits", "integration_splits_file"), ("timeline_records", "integration_timeline_file")]:
                dc[field] = str(root / FIXTURE[fixture])
            dc["issues_dir"] = str(root)
            dc_path = root / FIXTURE["integration_data_config_file"]
            dc_path.write_text(yaml.safe_dump(dc, allow_unicode=True))
            bc["data_config"] = str(dc_path)
            bc_path = root / FIXTURE["integration_b0_config_file"]
            bc_path.write_text(yaml.safe_dump(bc, allow_unicode=True))
            case = FIXTURE["timeline_cases"][0]
            vid = FIXTURE["fixture_video_id"]
            manifest = root / FIXTURE["integration_manifest_file"]
            output = root / FIXTURE["integration_output_dir"]
            write_jsonl(manifest, [{"video_id": vid, "category": FIXTURE["integration_category"],
                       "path": str(root / FIXTURE["integration_source_file"]), "duration_sec": case["expected"][-1],
                       "split": "test", "camera_group": FIXTURE["integration_group"]}])
            write_json(dc["splits"], {"train": [], "val": [], "test": [vid]})
            write_jsonl(dc["timeline_records"], [{"video_id": vid, "pts": case["pts"], "time_base": case["time_base"]}])
            frozen = types.ModuleType("src.model.holmesvau_infer")
            model = types.SimpleNamespace(language_model=types.SimpleNamespace(generate=Mock(return_value=[FIXTURE['a1_mock_token_ids']])))
            frozen.load_model = Mock(return_value=(model, object()))
            frozen.uniform_indices = Mock(return_value=FIXTURE["integration_frame_indices"])
            def fake_generate(path, prompt, model, tokenizer, **kw):
                model.language_model.generate(inputs_embeds=object(), max_new_tokens=kw['max_new_tokens'], eos_token_id=FIXTURE['a1_eos_token_id'])
                return FIXTURE["invalid_outputs"][0], FIXTURE["integration_frame_indices"]
            frozen.generate = Mock(side_effect=fake_generate)
            torch = types.ModuleType("torch")
            torch.set_num_threads = Mock()
            decord = types.ModuleType("decord")
            decord.cpu = Mock(return_value=None)
            decord.VideoReader = Mock(return_value=case["pts"])
            argv = ["b0", "--manifest", str(manifest), "--split", "test", "--out", str(output), "--config", str(bc_path), '--a1-config', FIXTURE['a1_config']]
            with patch.dict(sys.modules, {"torch": torch, "decord": decord, "src.model.holmesvau_infer": frozen}):
                with patch.object(sys, "argv", argv):
                    b0_main()
                with patch.object(sys, "argv", argv + ["--resume"]):
                    b0_main()
            frozen.generate.assert_called_once()
            frozen.load_model.assert_called_once()
            rows = read_jsonl(output / bc["predictions_file"])
            self.assertEqual(len(rows), len([vid]))
            self.assertFalse(rows[0]["parse_ok"])
            self.assertEqual(rows[0]["raw_output"], FIXTURE["invalid_outputs"][0])
            self.assertEqual(len(read_jsonl(output / bc["attempts_file"])), len([vid]))


def main():
    global FIXTURE
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    FIXTURE = config(args.config)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Spec04Tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    write_json(FIXTURE["test_results"], {"run": result.testsRun, "successful": result.wasSuccessful(),
               "failures": [(str(t), e) for t, e in result.failures], "errors": [(str(t), e) for t, e in result.errors],
               "test_names": unittest.defaultTestLoader.getTestCaseNames(Spec04Tests)})
    raise SystemExit(not result.wasSuccessful())


if __name__ == "__main__":
    main()
