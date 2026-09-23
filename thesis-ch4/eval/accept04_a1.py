"""One machine-checkable result per A1 §E item; B0 completeness stays FAIL."""
import argparse
import json
from pathlib import Path

from data.common import config, write_json
from eval.reparse_v2 import digest, load_inputs, protected_hashes, IntervalParser, legacy_evidence, measurement


def run(cfg, config_path):
    bc, originals, attempts, manifest, interrupted, diag, before = load_inputs(cfg)
    out = Path(cfg['output_dir'])
    predictions = json.loads((out / cfg['predictions_file']).read_text())
    rows = predictions['records']
    metrics = json.loads((out / cfg['metrics_file']).read_text())
    histogram = json.loads((out / cfg['histogram_file']).read_text())
    audit = json.loads((out / cfg['audit_file']).read_text())
    tests = json.loads(Path(cfg['tests_result']).read_text())
    parser = IntervalParser(cfg, bc['format_regex'])
    recomputed = []
    for r in originals:
        value = {k: r[k] for k in ('video_id', 'raw_output', 'frames_seen')}
        value.update(parser.parse(r['raw_output'], manifest[r['video_id']]['duration_sec']))
        value.update(legacy_evidence(r['raw_output'], diag[r['video_id']]['reencoded_tokens'], cfg))
        recomputed.append(value)
    recomputed_metrics = measurement(recomputed, manifest, cfg)
    checks = []
    def check(identifier, criterion, passed, evidence):
        checks.append({'id': identifier, 'criterion': criterion, 'passed': bool(passed),
                       'status': 'PASS' if passed else 'FAIL', 'evidence': evidence})

    same = rows == recomputed and all(metrics[k] == v for k, v in recomputed_metrics.items())
    check('E.1', '312 条复算及原因码直方图，逐条可重现',
          len(rows) == cfg['expected_persisted'] and same and
          histogram['layer_a'] == metrics['layer_a_reason_histogram'] and
          histogram['layer_b'] == metrics['layer_b_reason_histogram'] and
          sum(histogram['layer_a'].values()) + metrics['layer_a_pass_count'] == len(rows),
          {'layer_a_pass_count': metrics['layer_a_pass_count'], 'layer_a_failure_histogram': histogram['layer_a'],
           'layer_b_histogram': histogram['layer_b'], 'reproduced': same})
    after = protected_hashes(cfg)
    hashes = {p: {'before_sha256': before[p], 'after_sha256': after[p], 'unchanged': before[p] == after[p]}
              for p in (cfg['original_predictions'], cfg['original_metrics'])}
    check('E.2', '原 predictions/metrics SHA-256 与修订前相同', all(v['unchanged'] for v in hashes.values()), hashes)
    files = [out / cfg[k] for k in ('predictions_file', 'metrics_file', 'histogram_file', 'report_file', 'audit_file')]
    headers = {str(p): (p.read_text().startswith(cfg['scope_notice']) if p.suffix == '.md' else
                        list(json.loads(p.read_text()))[0] == 'scope_notice' and json.loads(p.read_text())['scope_notice'] == cfg['scope_notice']) for p in files}
    check('E.3', '各结果文件头声明顺序前缀、类别与不可外推', all(headers.values()), headers)
    check('E.4', 'format_ok_rate 仍为 0，严格格式定义不变', metrics['format_ok_rate'] == 0 and
          all(not r['format_ok'] for r in rows), {'format_ok_rate': metrics['format_ok_rate'], 'source_format_regex': bc['format_regex']})
    test_current = tests['test_config_sha256'] == digest(cfg['tests_config'])
    check('E.5', 'Layer A/B 每条允许和拒绝规则的单测', tests['successful'] and test_current and
          set(cfg['required_parser_tests']) <= set(tests['parser_cases']), {'cases': tests['parser_cases'], 'tests_successful': tests['successful']})
    check('E.6', 'stop_reason EOS→预算→other 三分支', tests['successful'] and test_current and
          set(cfg['required_stop_tests']) <= set(tests['test_names']), {'required_tests': cfg['required_stop_tests']})
    check('E.7', '312 条 stop_reason=unknown，绝不回填权威证据', len(rows) == cfg['expected_persisted'] and
          all(r['stop_reason'] == 'unknown' and r['generated_token_count'] is None and r['generated_token_ids'] is None
              and r['last_token_is_eos'] is None and r['truncated'] is None for r in rows),
          {'unknown_count': sum(r['stop_reason'] == 'unknown' for r in rows),
           'truncation_suspected_count': sum(r['truncation_suspected'] for r in rows), 'authoritative_backfill_count': 0})
    extras = {
        'all_protected_sha256_unchanged': before == after == audit['protected_before'] == audit['protected_after'],
        'spec_body_hash_matches': digest(cfg['spec']) == cfg['spec_sha256'],
        'result_configuration_matches': all(json.loads(p.read_text())['configuration_sha256'] == digest(config_path)
                                             for p in files if p.suffix == '.json'),
        'budget_unchanged': bc['max_new_tokens'] == cfg['max_new_tokens'],
        'ledger_consistent': metrics['n_attempted'] == len(attempts) and metrics['n_persisted'] == len(originals)
                             and metrics['n_interrupted'] == len(interrupted),
        'future_recording_integration_tests_passed': tests['successful'] and set(cfg['required_integration_tests']) <= set(tests['test_names']),
        'b0_stays_incomplete': metrics['complete'] is False and len(rows) < cfg['expected_test'],
    }
    passed = all(c['passed'] for c in checks) and all(extras.values())
    result = {'spec': 'SPEC-04-A1', 'status': 'passed' if passed else 'failed', 'implementation_complete': passed,
              'scope_notice': cfg['scope_notice'], 'checks': checks, 'additional_checks': extras,
              'n_attempted': len(attempts), 'n_persisted': len(originals), 'n_interrupted': len(interrupted),
              'interrupted': interrupted, 'complete': False,
              'b0_full_test_coverage': {'status': 'FAIL', 'persisted': len(rows), 'expected': cfg['expected_test']},
              'protected_files': {p: {'before_sha256': before[p], 'after_sha256': after[p], 'unchanged': before[p] == after[p]} for p in before},
              'tests': tests, 'model_generation_calls': 0,
              'issues': sorted(str(p) for p in Path(cfg['issue_dir']).glob('04a1-*.md'))}
    write_json(cfg['acceptance'], result)
    print(json.dumps({'status': result['status'], 'passed': sum(c['passed'] for c in checks),
                      'total': len(checks), 'additional_checks': extras, 'b0_full_test_coverage': result['b0_full_test_coverage']}, ensure_ascii=False))
    return passed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()
    raise SystemExit(not run(config(args.config), args.config))


if __name__ == '__main__':
    main()
