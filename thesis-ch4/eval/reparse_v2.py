"""SPEC-04-A1 offline measurement revision. No model imports or generation calls."""
import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import statistics

from data.common import config, read_jsonl, write_json, write_text
from eval.b0_selfdata import summarize


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected_hashes(cfg):
    baseline = json.loads(Path(cfg['baseline']).read_text())['protected_sha256']
    current = {p: digest(p) for p in baseline}
    if baseline != current:
        raise RuntimeError('Protected input changed: ' + str([p for p in baseline if baseline[p] != current[p]]))
    return current


class IntervalParser:
    """Parse only the declared grammar, retaining all evidence of ambiguity."""
    def __init__(self, cfg, format_regex):
        self.cfg, self.grammar = cfg, cfg['grammar']
        g = self.grammar
        values = dict(number=g['number'], left=g['left_boundary'], right=g['right_boundary'], unit=g['unit'])
        self.patterns = {name: re.compile(pattern.format(**values)) for name, pattern in g['patterns'].items()}
        self.incomplete = [re.compile(p.format(**values)) for p in g['missing_unit_patterns']]
        self.points = re.compile(g['time_value_pattern'].format(**values))
        self.clock = re.compile(g['clock_pattern'])
        self.format = re.compile(format_regex)

    def parse(self, raw, duration_sec):
        matches = []
        # A match is an occurrence, not a deduplicated numeric value: repeated
        # identical intervals remain multi_interval. Deduplicate only same spans.
        seen = set()
        for family, pattern in self.patterns.items():
            for m in pattern.finditer(raw):
                key = (m.start('start'), m.end('start'), m.start('end'), m.end('end'))
                if key not in seen:
                    matches.append((family, m))
                    seen.add(key)
        matches.sort(key=lambda pair: pair[1].start())
        spans = [m.span() for _, m in matches]
        incomplete = [m for p in self.incomplete for m in p.finditer(raw)
                      if not any(a <= m.start() and m.end() <= b for a, b in spans)]
        failures = set()
        if self.clock.search(raw):
            failures.add('ambiguous_clock')
        if len(matches) > 1:
            failures.add('multi_interval')
        if incomplete:
            failures.add('missing_unit')
        if not matches and not failures:
            failures.add('single_point' if len(list(self.points.finditer(raw))) == 1 else 'no_interval')
        reasons = [r for r in self.grammar['failure_priority'] if r in failures]
        ok = len(matches) == 1 and not reasons
        result = {'parse_ok': ok, 'layer_a_reason': None if ok else reasons[0],
                  'layer_a_failure_reasons': reasons, 'layer_b_reason': None,
                  'layer_b_valid': None, 'parsed_start': None, 'parsed_end': None,
                  'out_of_range': None, 'format_ok': bool(self.format.fullmatch(raw)),
                  'matched_families': [family for family, _ in matches]}
        if ok:
            m = matches[0][1]
            start, end = Decimal(m['start']), Decimal(m['end'])
            b = 'reverse_order' if end < start else 'zero_duration' if end == start else 'valid'
            result.update(parsed_start=float(start), parsed_end=float(end), layer_b_reason=b,
                          layer_b_valid=end >= start, out_of_range=end > Decimal(str(duration_sec)))
        return result


def legacy_evidence(raw, reencoded_length, cfg):
    """Unknown is deliberately not filled from decoded/re-encoded text."""
    return {'stop_reason': 'unknown', 'generated_token_count': None,
            'generated_token_ids': None, 'last_token_is_eos': None,
            'max_new_tokens': cfg['max_new_tokens'], 'truncated': None,
            'retokenized_len': reencoded_length,
            'truncation_suspected': reencoded_length == cfg['max_new_tokens'] and
                                    not raw.endswith(tuple(cfg['sentence_endings']))}


def measurement(records, manifest, cfg):
    n = len(records)
    passed = [r for r in records if r['parse_ok']]
    valid = [r for r in passed if r['layer_b_reason'] in cfg['grammar']['distribution_layer_b']]
    a_hist = {k: sum(r['layer_a_reason'] == k for r in records) for k in cfg['grammar']['layer_a_failure_codes']}
    b_hist = {k: sum(r['layer_b_reason'] == k for r in passed) for k in cfg['grammar']['layer_b_codes']}
    full = sum(abs(r['parsed_start']) <= cfg['full_clip_tolerance_sec'] and
               abs(r['parsed_end'] - manifest[r['video_id']]['duration_sec']) <= cfg['full_clip_tolerance_sec']
               for r in valid)
    return {'sample_count': n, 'expected_count': cfg['expected_test'], 'complete': n == cfg['expected_test'],
            'layer_a_pass_count': len(passed), 'layer_a_fail_count': n - len(passed),
            'parse_fail_rate': 1 - len(passed) / n if n else None,
            'format_ok_rate': sum(r['format_ok'] for r in records) / n if n else None,
            'layer_a_reason_histogram': a_hist, 'layer_b_reason_histogram': b_hist,
            'out_of_range_count': sum(bool(r['out_of_range']) for r in passed),
            'duration_distribution_sec': summarize([r['parsed_end'] - r['parsed_start'] for r in valid], cfg),
            'start_distribution_sec': summarize([r['parsed_start'] for r in valid], cfg),
            'reverse_order_intervals': [{'video_id': r['video_id'], 'start': r['parsed_start'], 'end': r['parsed_end']}
                                        for r in passed if r['layer_b_reason'] == 'reverse_order'],
            'avg_frames': statistics.mean(r['frames_seen'] for r in records) if n else None,
            'full_clip_ratio': full / n if n else None,
            'truncation_suspected_count': sum(r['truncation_suspected'] for r in records if 'truncation_suspected' in r),
            'rate_denominator': 'persisted records only; interrupted attempts excluded',
            'distribution_population': 'Layer A passed and Layer B valid/zero_duration; reverse_order listed separately'}


def load_inputs(cfg):
    before = protected_hashes(cfg)
    if digest(cfg['spec']) != cfg['spec_sha256']:
        raise RuntimeError('Frozen A1 body changed')
    original_cfg = config(cfg['original_config'])
    if original_cfg['max_new_tokens'] != cfg['max_new_tokens']:
        raise RuntimeError('Generation budget conflict')
    records = read_jsonl(cfg['original_predictions'])
    attempts = read_jsonl(cfg['attempts'])
    manifest = {r['video_id']: r for r in read_jsonl(cfg['manifest'])}
    splits = json.loads(Path(cfg['splits']).read_text())
    test = splits['test']
    ids = [r['video_id'] for r in records]
    attempted_ids = [r['video_id'] for r in attempts]
    interrupted = [v for v in attempted_ids if v not in set(ids)]
    categories = Counter(manifest[v]['category'] for v in ids)
    if (len(ids) != cfg['expected_persisted'] or len(attempts) != cfg['expected_attempted'] or
        len(interrupted) != cfg['expected_interrupted'] or len(test) != cfg['expected_test'] or
        categories != cfg['expected_categories'] or len(set(ids)) != len(ids) or
        len(set(attempted_ids)) != len(attempts) or not set(ids) <= set(attempted_ids) or
        not set(attempted_ids) <= set(test) or attempted_ids[:len(ids)] != ids):
        raise RuntimeError('Frozen cohort or ledger conflict')
    diagnosis = json.loads(Path(cfg['diagnosis']).read_text())
    for p in (cfg['original_predictions'], cfg['original_config'], cfg['manifest']):
        if diagnosis['source_sha256'][p] != digest(p):
            raise RuntimeError('Diagnostic token-length cache no longer matches input')
    evidence = {r['video_id']: r for r in diagnosis['evidence']}
    if set(evidence) != set(ids) or any(evidence[r['video_id']]['raw_output'] != r['raw_output'] for r in records):
        raise RuntimeError('Cached tokenizer evidence mismatch')
    return original_cfg, records, attempts, manifest, interrupted, evidence, before


def run(cfg, config_path):
    original_cfg, records, attempts, manifest, interrupted, evidence, before = load_inputs(cfg)
    parser = IntervalParser(cfg, original_cfg['format_regex'])
    output = Path(cfg['output_dir'])
    # Never route new results into any immutable input directory.
    if any(output.resolve() == Path(p).resolve().parent or output.resolve() == Path(p).resolve()
           for p in before):
        raise RuntimeError('Output destination overlaps protected inputs')
    parsed = []
    for row in records:
        vid, raw = row['video_id'], row['raw_output']
        result = {k: row[k] for k in ('video_id', 'raw_output', 'frames_seen')}
        result.update(parser.parse(raw, manifest[vid]['duration_sec']))
        result.update(legacy_evidence(raw, evidence[vid]['reencoded_tokens'], cfg))
        parsed.append(result)
        if len(parsed) == cfg['monitor']['first_batch'] or len(parsed) % cfg['monitor']['every'] == 0:
            print(json.dumps({'phase': 'offline_checkpoint', 'n': len(parsed),
                              'layer_a_pass': sum(r['parse_ok'] for r in parsed),
                              'last_raw': raw[:cfg['monitor']['preview_chars']]}, ensure_ascii=False), flush=True)
    header = {'scope_notice': cfg['scope_notice'], 'measurement_version': 'SPEC-04-A1/reparse_v2',
              'configuration_sha256': digest(config_path), 'spec_sha256': digest(cfg['spec']),
              'model_generation_calls': 0}
    ledger = {'n_attempted': len(attempts), 'n_persisted': len(records), 'n_interrupted': len(interrupted),
              'interrupted': [{'video_id': v, 'status': 'interrupted', 'counted_as_parse_failure': False} for v in interrupted]}
    result = {**header, **measurement(parsed, manifest, cfg), **ledger}
    write_json(output / cfg['predictions_file'], {**header, 'complete': False, 'records': parsed})
    write_json(output / cfg['metrics_file'], result)
    write_json(output / cfg['histogram_file'], {**header,
               'layer_a': result['layer_a_reason_histogram'], 'layer_b': result['layer_b_reason_histogram'],
               'layer_a_pass_count': result['layer_a_pass_count'],
               'note': 'Layer A primary reasons are exclusive; all detected reasons remain on each record.'})
    after = protected_hashes(cfg)
    write_json(output / cfg['audit_file'], {**header, 'protected_before': before, 'protected_after': after,
               'unchanged': before == after, 'retokenized_length_source': cfg['diagnosis'],
               'retokenized_length_source_sha256': digest(cfg['diagnosis']), **ledger})
    write_text(output / cfg['report_file'], cfg['scope_notice'] + '\n\n# SPEC-04-A1 离线复算\n\n' +
               'Layer A 以 A1 的“恰好一对无歧义非负秒数”为条件，覆盖明列的四类语法；同时实现两端都显式标秒的范围及 start/end respectively 表达。验收不设约 188 的目标值。\n\n' +
               json.dumps(result, ensure_ascii=False, indent=2) + '\n\n' +
               '原文未改；逆序区间单列，不交换、不进合法区间分布；零时长和越界值不裁剪。\n' +
               '存量停止原因均为 unknown；truncated=null 表示没有权威证据，启发式仅记 suspected。\n')
    print(json.dumps({'layer_a_pass_count': result['layer_a_pass_count'], 'parse_fail_rate': result['parse_fail_rate'],
                      'histogram': result['layer_a_reason_histogram'], **ledger}, ensure_ascii=False), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = config(args.config)
    try:
        run(cfg, args.config)
    except Exception as exc:
        write_text(Path(cfg['issue_dir']) / cfg['issue_name'], '# SPEC-04-A1 execution conflict\n\n' + repr(exc) + '\nStopped; no model generation performed.\n')
        raise


if __name__ == '__main__':
    main()
