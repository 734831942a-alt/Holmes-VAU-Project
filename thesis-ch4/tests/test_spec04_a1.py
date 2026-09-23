"""A1 contract tests use handcrafted text/token IDs; never call a real model."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import yaml
import numpy as np  # Load before isolated sys.modules mocking; avoid partial reload.

from data.common import config, read_jsonl, write_json, write_jsonl
from eval.generation_evidence import stopping_evidence, capture_generation, generate_with_evidence
from eval.reparse_v2 import IntervalParser, legacy_evidence, measurement, load_inputs, protected_hashes

F = None
C = None


class Spec04A1Tests(unittest.TestCase):
    def parser(self):
        return IntervalParser(C, config(C['original_config'])['format_regex'])

    def test_every_layer_a_and_b_rule(self):
        p = self.parser()
        for case in F['parser_cases']:
            with self.subTest(rule=case['name']):
                result = p.parse(case['raw'], F['duration_sec'])
                self.assertEqual(result['parse_ok'], case['ok'])
                if 'reason' in case:
                    self.assertEqual(result['layer_a_reason'], case['reason'])
                if case['ok']:
                    self.assertEqual(result['layer_b_reason'], case['b'])
                    self.assertEqual((result['parsed_start'], result['parsed_end']), (case['start'], case['end']))
                    self.assertEqual(result['out_of_range'], case.get('outside', False))
                    self.assertEqual(result['format_ok'], case['format'])

    def test_zero_outside_in_distributions_reverse_separate(self):
        selected = [c for c in F['parser_cases'] if c['name'] in ('zero_duration', 'out_of_range', 'reverse_order')]
        records = [dict(video_id=F['fixture_video_id'], frames_seen=F['frames_seen'], **self.parser().parse(c['raw'], F['duration_sec'])) for c in selected]
        m = measurement(records, {F['fixture_video_id']: {'duration_sec': F['duration_sec']}}, C)
        self.assertEqual(m['parse_fail_rate'], 0)
        self.assertEqual(m['duration_distribution_sec']['min'], 0)
        self.assertEqual(m['duration_distribution_sec']['count'], len(selected) - len(m['reverse_order_intervals']))
        self.assertGreater(m['duration_distribution_sec']['max'], F['duration_sec'])
        self.assertEqual(m['reverse_order_intervals'][0]['start'], next(c for c in selected if c['b'] == 'reverse_order')['start'])

    def test_stop_reason_eos_wins_at_budget_and_before_last(self):
        ids = [F['ordinary_token_id']] * C['max_new_tokens']
        ids[0] = F['eos_token_id']
        r = stopping_evidence(ids, [], F['eos_token_id'], C['max_new_tokens'])
        self.assertEqual(r['stop_reason'], 'eos')
        self.assertFalse(r['truncated'])
        self.assertFalse(r['last_token_is_eos'])
        ids[-1] = F['eos_token_id']
        self.assertTrue(stopping_evidence(ids, [], F['eos_token_id'], C['max_new_tokens'])['last_token_is_eos'])

    def test_stop_reason_max_new_tokens(self):
        ids = [F['ordinary_token_id']] * C['max_new_tokens']
        r = stopping_evidence(ids, [], F['eos_token_id'], C['max_new_tokens'])
        self.assertEqual(r['stop_reason'], 'max_new_tokens')
        self.assertTrue(r['truncated'])
        self.assertEqual(r['generated_token_ids'], ids)

    def test_stop_reason_other_and_prefix_formula(self):
        ids, prefix = F['other_tokens'], F['input_prefix']
        r = stopping_evidence(prefix + ids, prefix, F['eos_token_id'], C['max_new_tokens'])
        self.assertEqual(r['stop_reason'], 'other')
        self.assertEqual(r['generated_token_count'], len(ids))
        self.assertEqual(r['generated_token_ids'], ids)
        self.assertFalse(r['truncated'])
        with self.assertRaises(ValueError):
            stopping_evidence(ids, prefix, F['eos_token_id'], C['max_new_tokens'])

    def test_capture_forwards_actual_return_and_restores_on_error(self):
        ids = F['other_tokens'] + [F['eos_token_id']]
        lm = types.SimpleNamespace(generate=Mock(return_value=[ids]))
        original = lm.generate
        kwargs = {'inputs_embeds': object(), 'max_new_tokens': C['max_new_tokens'], 'eos_token_id': F['eos_token_id']}
        with capture_generation(lm, C['max_new_tokens']) as evidence:
            self.assertEqual(lm.generate(**kwargs), [ids])
        original.assert_called_once_with(**kwargs)
        self.assertIs(lm.generate, original)
        self.assertEqual(evidence[0]['input_prefix_token_count'], 0)
        self.assertEqual(evidence[0]['generated_token_count'], len(ids))
        with self.assertRaises(RuntimeError):
            with capture_generation(lm, C['max_new_tokens']):
                raise RuntimeError('handcrafted exception')
        self.assertIs(lm.generate, original)

    def test_wrapper_returns_unchanged_text_and_never_reencodes(self):
        ids = F['other_tokens'] + [F['eos_token_id']]
        model = types.SimpleNamespace(language_model=types.SimpleNamespace(generate=Mock(return_value=[ids])))
        tokenizer = Mock()
        tokenizer.encode.side_effect = AssertionError('must not reencode')
        def fake_generate(path, prompt, actual_model, tok, **kw):
            actual_model.language_model.generate(inputs_embeds=object(), max_new_tokens=kw['max_new_tokens'], eos_token_id=F['eos_token_id'])
            return F['mock_raw_output'], F['mock_frames']
        raw, frames, evidence = generate_with_evidence(F['source_path'], '', model, tokenizer,
                generate_fn=fake_generate, max_new_tokens=C['max_new_tokens'])
        self.assertEqual(raw, F['mock_raw_output'])
        self.assertEqual(frames, F['mock_frames'])
        self.assertEqual(evidence['generated_token_ids'], ids)
        tokenizer.encode.assert_not_called()

    def test_capture_rejects_second_call_without_retry(self):
        lm = types.SimpleNamespace(generate=Mock(return_value=[F['other_tokens']]))
        original = lm.generate
        kw = {'inputs_embeds': object(), 'max_new_tokens': C['max_new_tokens'], 'eos_token_id': F['eos_token_id']}
        with capture_generation(lm, C['max_new_tokens']):
            lm.generate(**kw)
            with self.assertRaisesRegex(RuntimeError, 'More than one'):
                lm.generate(**kw)
        original.assert_called_once()

    def test_original_b0_directory_is_rejected_before_loading_model(self):
        from eval.b0_selfdata import main as b0_main
        before = protected_hashes(C)
        argv = ['b0', '--manifest', C['manifest'], '--split', 'test', '--out', str(Path(C['original_predictions']).parent),
                '--config', C['original_config'], '--a1-config', F['reparse_config'], '--resume']
        with patch.object(sys, 'argv', argv):
            with self.assertRaisesRegex(RuntimeError, 'immutable'):
                b0_main()
        self.assertEqual(before, protected_hashes(C))

    def test_legacy_unknown_and_suspected_are_separate(self):
        r = legacy_evidence(F['suspected_text'], C['max_new_tokens'], C)
        self.assertEqual(r['stop_reason'], 'unknown')
        self.assertTrue(r['truncation_suspected'])
        self.assertIsNone(r['truncated'])
        self.assertIsNone(r['generated_token_ids'])
        self.assertFalse(legacy_evidence(F['completed_text'], C['max_new_tokens'], C)['truncation_suspected'])
        self.assertFalse(legacy_evidence(F['suspected_text'], C['max_new_tokens'] - 1, C)['truncation_suspected'])

    def test_all_312_records_unknown_and_original_metrics_hash_unchanged(self):
        before = protected_hashes(C)
        _, records, attempts, manifest, interrupted, evidence, _ = load_inputs(C)
        parsed = []
        for r in records:
            parsed.append(legacy_evidence(r['raw_output'], evidence[r['video_id']]['reencoded_tokens'], C))
        self.assertEqual(len(parsed), C['expected_persisted'])
        self.assertTrue(all(r['stop_reason'] == 'unknown' and r['generated_token_count'] is None for r in parsed))
        self.assertEqual(len(attempts) - len(records), len(interrupted))
        self.assertEqual(len(interrupted), C['expected_interrupted'])
        self.assertEqual(protected_hashes(C), before)
        self.assertEqual(hashlib.sha256(Path(F['original_metrics']).read_bytes()).hexdigest(), before[F['original_metrics']])

    def run_future_case(self, count, should_pause=False):
        from eval.b0_selfdata import main as b0_main
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bc = config(C['original_config'])
            dc = config(bc['data_config'])
            dc.update(splits=str(root / F['fixture_splits']), timeline_records=str(root / F['fixture_timeline']), issues_dir=str(root))
            dp = root / F['fixture_future_data_config']
            dp.write_text(yaml.safe_dump(dc, allow_unicode=True))
            bc['data_config'] = str(dp)
            bp = root / F['fixture_future_config']
            bp.write_text(yaml.safe_dump(bc, allow_unicode=True))
            manifest, out = root / F['fixture_manifest'], root / F['fixture_output']
            vid = F['fixture_video_id']
            vids = [vid + '-' + str(i) for i in range(count)]
            write_jsonl(manifest, [{'video_id': v, 'category': F['category'], 'path': F['source_path'],
                        'split': 'test', 'camera_group': F['camera_group'], 'duration_sec': F['duration_sec']} for v in vids])
            write_json(dc['splits'], {'train': [], 'val': [], 'test': vids})
            write_jsonl(dc['timeline_records'], [{'video_id': v, 'pts': F['timeline_pts'], 'time_base': F['time_base']} for v in vids])
            ids = F['other_tokens'] + [F['eos_token_id']]
            lm = types.SimpleNamespace(generate=Mock(return_value=[ids]))
            model = types.SimpleNamespace(language_model=lm)
            frozen = types.ModuleType('src.model.holmesvau_infer')
            frozen.load_model = Mock(return_value=(model, object()))
            frozen.uniform_indices = Mock(return_value=F['mock_frames'])
            def fake_generate(path, prompt, model, tok, **kw):
                model.language_model.generate(inputs_embeds=object(), max_new_tokens=kw['max_new_tokens'], eos_token_id=F['eos_token_id'])
                return F['no_interval_text'], F['mock_frames']
            frozen.generate = Mock(side_effect=fake_generate)
            torch = types.ModuleType('torch'); torch.set_num_threads = Mock()
            decord = types.ModuleType('decord'); decord.cpu = Mock(); decord.VideoReader = Mock(return_value=F['timeline_pts'])
            argv = ['b0', '--manifest', str(manifest), '--split', 'test', '--out', str(out), '--config', str(bp), '--a1-config', F['reparse_config']]
            with patch.dict(sys.modules, {'torch': torch, 'decord': decord, 'src.model.holmesvau_infer': frozen}):
                if should_pause:
                    with patch.object(sys, 'argv', argv):
                        with self.assertRaisesRegex(RuntimeError, 'monitoring pause'):
                            b0_main()
                else:
                    with patch.object(sys, 'argv', argv): b0_main()
                    with patch.object(sys, 'argv', argv + ['--resume']): b0_main()
            expected = C['future_monitor']['first_batch'] if should_pause else count
            self.assertEqual(frozen.generate.call_count, expected)
            frozen.load_model.assert_called_once()
            rows, ledger = read_jsonl(out / bc['predictions_file']), read_jsonl(out / bc['attempts_file'])
            self.assertEqual(len(rows), expected)
            self.assertEqual(len(ledger), expected)
            self.assertFalse(rows[0]['parse_ok'])
            self.assertEqual(rows[0]['raw_output'], F['no_interval_text'])
            for name in ('stop_reason', 'generated_token_count', 'max_new_tokens', 'last_token_is_eos', 'generated_token_ids'):
                self.assertEqual(rows[0][name], ledger[0][name])
            self.assertEqual(rows[0]['stop_reason'], 'eos')
            self.assertEqual(rows[0]['generated_token_ids'], ids)

    def test_future_cli_persists_evidence_and_never_retries(self):
        self.run_future_case(1)

    def test_future_monitor_stops_after_small_batch(self):
        self.run_future_case(C['future_monitor']['first_batch'] + 1, should_pause=True)


def main():
    global F, C
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    F = config(args.config)
    C = config(F['reparse_config'])
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Spec04A1Tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    write_json(F['test_results'], {'run': result.testsRun, 'successful': result.wasSuccessful(),
        'test_names': unittest.defaultTestLoader.getTestCaseNames(Spec04A1Tests),
        'parser_cases': [c['name'] for c in F['parser_cases']],
        'test_config_sha256': hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        'failures': [(str(t), e) for t, e in result.failures], 'errors': [(str(t), e) for t, e in result.errors],
        'real_model_generations': 0})
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
