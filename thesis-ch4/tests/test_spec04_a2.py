"""No real model calls: A2 monitoring, immutable inputs, and durable resume."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import numpy as np
import yaml

from data.common import config, write_json, write_jsonl, read_jsonl
from eval import b0_full_test as full
from eval.generation_evidence import stopping_evidence

F = C = None


class A2Tests(unittest.TestCase):
    def baseline_rates(self):
        return {k: 0 for k in C['monitor_thresholds']}

    def record(self, vid):
        bc = config(C['generation_config'])
        return {'video_id': vid, 'raw_output': F['raw_output'], 'frames_seen': F['frames_seen'],
                'input_prefix_source': 'inputs_embeds_only_empty_token_prefix',
                **stopping_evidence(F['short_ids'], [], F['eos_token'], bc['max_new_tokens'])}

    def test_exact_frozen_thresholds_and_model_behavior_never_pauses(self):
        rates = self.baseline_rates()
        rates.update(parse_fail_rate=1, format_ok_rate=0, no_interval_rate=1)
        rates['empty_raw_output_rate'] = F['below_empty_rate']
        rates['truncated_rate'] = F['below_truncated_rate']
        self.assertEqual(full.pause_signals(rates, C), [])
        for key in ('generation_exception_rate', 'evidence_malformed_rate'):
            r = self.baseline_rates(); r[key] = F['positive_error_rate']
            self.assertEqual(full.pause_signals(r, C), [key])
        for key in ('empty_raw_output_rate', 'truncated_rate'):
            r = self.baseline_rates(); r[key] = C['monitor_thresholds'][key]['value']
            self.assertEqual(full.pause_signals(r, C), [key])

    def test_authoritative_evidence_validation_and_eos_precedence(self):
        bc = config(C['generation_config'])
        r = self.record(F['mock_video_prefix'])
        self.assertTrue(full.evidence_valid(r, bc, C))
        bad = copy.deepcopy(r); bad['generated_token_count'] += 1
        self.assertFalse(full.evidence_valid(bad, bc, C))
        bad = copy.deepcopy(r); bad.pop('generated_token_ids')
        self.assertFalse(full.evidence_valid(bad, bc, C))
        bad = copy.deepcopy(r); bad['stop_reason'] = 'unknown'
        self.assertFalse(full.evidence_valid(bad, bc, C))
        ids = [F['ordinary_token']] * bc['max_new_tokens']; ids[0] = F['eos_token']
        r.update(stopping_evidence(ids, [], F['eos_token'], bc['max_new_tokens']))
        self.assertEqual(r['stop_reason'], 'eos')
        self.assertTrue(full.evidence_valid(r, bc, C))

    def test_round_robin_keeps_exact_test_and_warmup_crosses_categories(self):
        bc = config(C['generation_config']); dc = config(bc['data_config'])
        rows = full.ordered_rows(read_jsonl(dc['manifest']), full.read_json(dc['splits'])['test'])
        self.assertEqual(len(rows), C['expected_test'])
        self.assertEqual(len({r['video_id'] for r in rows}), C['expected_test'])
        self.assertGreaterEqual(len({r['category'] for r in rows[:C['warmup_n']]}), C['warmup_min_categories'])
        checkpoints = list(range(C['warmup_n'], len(rows), C['checkpoint_every'])) + [len(rows)]
        self.assertEqual(checkpoints, F['expected_checkpoints'])

    def test_checkpoint_requires_eight_raw_and_ignores_parse_failure(self):
        with tempfile.TemporaryDirectory() as d:
            c = copy.deepcopy(C)
            c['measurement_dir'] = str(Path(d)/F['fixture_names']['measurement_dir'])
            c['generation_dir'] = str(Path(d)/F['fixture_names']['generation_dir'])
            c['issue_dir'] = str(Path(d)/F['fixture_names']['issues'])
            records = [self.record(F['mock_video_prefix']+str(i)) for i in range(C['warmup_n'])]
            rows = [{'video_id': r['video_id'], 'duration_sec': F['duration_sec']} for r in records]
            cp = full.make_checkpoint(c, records, config(C['generation_config']), config(C['measurement_config']), rows, 0)
            self.assertEqual(len(cp['raw_samples']), C['raw_sample_on_pause'])
            self.assertTrue(all(not r['a1_parse']['parse_ok'] for r in cp['raw_samples']))
            full.review(c, len(records), 'continue', 'Mock review: eight nonempty no_interval outputs, valid EOS evidence.')
            review = full.read_json(full.review_path(c, len(records)))
            self.assertEqual(review['signals_after_raw_review'], [])

    def test_clean_interruption_resumes_without_regenerating_persisted_rows(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); c = copy.deepcopy(C)
            names = F['fixture_names']
            c.update(generation_dir=str(root/names['generation_dir']), measurement_dir=str(root/names['measurement_dir']), issue_dir=str(root/names['issues']), expected_test=F['mock_count'], source_files=[])
            bc = config(C['generation_config']); ac = config(C['measurement_config']); dc = config(bc['data_config'])
            dc['timeline_records'] = str(root/names['timeline'])
            bcpath = root/names['generation_config']; bcpath.write_text(yaml.safe_dump(bc)); c['generation_config'] = str(bcpath)
            cfgpath = root/names['configuration']; cfgpath.write_text(yaml.safe_dump(c))
            rows = [{'video_id': F['mock_video_prefix']+str(i), 'category': F['category_fixture'][i % len(F['category_fixture'])],
                     'path': str(root/names['raw_video']), 'duration_sec': F['duration_sec']} for i in range(F['mock_count'])]
            write_jsonl(dc['timeline_records'], [{'video_id': r['video_id'], 'pts': F['timeline_pts'], 'time_base': F['time_base']} for r in rows])
            run = {'run_id': 'mock', 'config_sha256': full.sha(cfgpath), 'generation_config': bc,
                   'generation_config_sha256': full.sha(bcpath), 'order': [r['video_id'] for r in rows], 'source_sha256': {}}
            write_json(full.artifact(c,'run',True), run)
            frozen = types.ModuleType('src.model.holmesvau_infer')
            lm = types.SimpleNamespace(generate=Mock(return_value=[F['short_ids']]))
            model = types.SimpleNamespace(language_model=lm)
            frozen.load_model = Mock(return_value=(model, object()))
            frozen.uniform_indices = Mock(return_value=F['mock_indices'])
            def fake_generate(path, prompt, model, tokenizer, **kw):
                model.language_model.generate(inputs_embeds=object(), max_new_tokens=kw['max_new_tokens'], eos_token_id=F['eos_token'])
                return F['raw_output'], F['mock_indices']
            frozen.generate = Mock(side_effect=fake_generate)
            torch = types.ModuleType('torch'); torch.set_num_threads = Mock()
            decord = types.ModuleType('decord'); decord.cpu = Mock(); decord.VideoReader = Mock(return_value=F['timeline_pts'])
            def acknowledge(cfg, cp, total):
                full.review(cfg, cp['n_generated'], 'continue', 'Mock review of eight raw outputs and actual mocked token evidence.')
            with patch.dict(sys.modules, {'torch':torch, 'decord':decord, 'src.model.holmesvau_infer':frozen}), \
                 patch.object(full,'load_context',return_value=(bc,ac,dc,rows)), \
                 patch.object(full,'audit',return_value={'unchanged':True,'sha256':{},'changed':[]}):
                with patch.object(full,'wait_for_review',side_effect=RuntimeError('simulated interruption at durable checkpoint')):
                    with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                        full.worker(c,str(cfgpath))
                self.assertEqual(len(read_jsonl(full.artifact(c,'predictions',True))), c['warmup_n'])
                with patch.object(full,'wait_for_review',side_effect=acknowledge):
                    full.worker(c,str(cfgpath),resume=True)
                self.assertEqual(frozen.generate.call_count, len(rows))
                full.worker(c,str(cfgpath),resume=True)
                self.assertEqual(frozen.generate.call_count, len(rows))
            self.assertEqual(len({r['video_id'] for r in read_jsonl(full.artifact(c,'attempts',True))}), len(rows))
            self.assertTrue(all(full.evidence_valid(r,bc,c) for r in read_jsonl(full.artifact(c,'predictions',True))))


def main():
    global F,C
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args()
    F=config(args.config);C=config(F['run_config'])
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(A2Tests))
    write_json(F['test_results'], {'run':result.testsRun,'successful':result.wasSuccessful(),
        'test_names':unittest.defaultTestLoader.getTestCaseNames(A2Tests),
        'failures':[(str(t),e) for t,e in result.failures], 'errors':[(str(t),e) for t,e in result.errors],
        'real_model_generations':0})
    raise SystemExit(not result.wasSuccessful())


if __name__=='__main__': main()
