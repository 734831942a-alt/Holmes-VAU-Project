"""Read-only audit of saved B0 text; never changes predictions or B0 metrics.

Diagnostic patterns are explicitly post-hoc descriptions, not a new B0 parser.
Re-encoding decoded text estimates output length; original token IDs/EOS were
not saved, so it cannot conclusively identify the generation stop reason.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from data.common import config, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    dc = config(args.config)
    bc = config(dc['b0_config'])
    paths = [dc[k] for k in ('b0_config', 'manifest', 'predictions', 'attempts', 'metrics')]
    def hashes():
        return {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    before = hashes()
    rows = read_jsonl(dc['predictions'])
    attempts = read_jsonl(dc['attempts'])
    manifest = {r['video_id']: r for r in read_jsonl(dc['manifest'])}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(bc['model_path'], trust_remote_code=True,
                                             use_fast=False, local_files_only=True)
    patterns = {k: re.compile(v) for k, v in dc['patterns'].items()}
    counts, lengths, categories = Counter(), Counter(), Counter()
    evidence = []
    for row in rows:
        raw = row['raw_output']
        length = len(tokenizer.encode(raw, add_special_tokens=False))
        lengths[length] += 1
        categories[manifest[row['video_id']]['category']] += 1
        matched = [(name, pat.fullmatch(raw)) for name, pat in patterns.items()]
        matched = [(name, m) for name, m in matched if m]
        item = {'video_id': row['video_id'], 'raw_output': raw,
                'official_parse_ok': row['parse_ok'], 'reencoded_tokens': length}
        if len(matched) == 1:
            form, match = matched[0]
            start, end = float(match['start']), float(match['end'])
            valid = 0 <= start < end
            counts[form] += 1
            if form in dc['unambiguous_seconds_forms'] and valid:
                counts['explicit_valid_seconds_pair'] += 1
            if not valid:
                counts['invalid_interval_order_or_zero_length'] += 1
            outside = start < 0 or end > manifest[row['video_id']]['duration_sec']
            counts['candidate_outside_video'] += outside
            item.update(form=form, diagnostic_start=start, diagnostic_end=end,
                        ordered_nonnegative=valid, outside_video=outside)
        elif re.search(dc['clock_pattern'], raw):
            counts['ambiguous_clock_not_converted'] += 1
            item['form'] = 'ambiguous_clock_not_converted'
        elif re.search(dc['single_time_pattern'], raw):
            counts['single_time_not_interval'] += 1
            item['form'] = 'single_time_not_interval'
        else:
            counts['no_explicit_interval'] += 1
            item['form'] = 'no_explicit_interval'
        counts['official_parse_failure'] += not row['parse_ok']
        counts['official_format_ok'] += row['format_ok']
        at_budget = length >= bc['max_new_tokens']
        unfinished = not raw.endswith(tuple(dc['sentence_endings']))
        counts['reencoded_at_or_above_budget'] += at_budget
        counts['budget_length_and_unfinished_sentence'] += at_budget and unfinished
        item.update(reencoded_at_or_above_budget=at_budget, unfinished_sentence=unfinished)
        evidence.append(item)
    persisted = {r['video_id'] for r in rows}
    active = []
    for path in Path(dc['proc_root']).iterdir():
        if path.name.isdigit():
            try:
                argv = (path / dc['proc_cmdline']).read_bytes().split(dc['proc_separator'].encode())
                if dc['process_module'].encode() in argv:
                    active.append(path.name)
            except OSError:
                pass
    after = hashes()
    if before != after:
        raise RuntimeError('B0 inputs changed during offline audit')
    result = {'scope': 'post-hoc diagnosis only; not replacement B0 metrics or evaluation',
              'model_generations_performed': 0, 'b0_files_unchanged': True,
              'source_sha256': before, 'sample_count': len(rows),
              'expected_test_count': sum(r['split'] == 'test' for r in manifest.values()),
              'active_b0_pids': active, 'counts': dict(counts),
              'category_counts': dict(categories), 'reencoded_token_lengths': dict(lengths),
              'max_new_tokens': bc['max_new_tokens'],
              'truncation_limitation': 'Re-encoded length only; generated token IDs and stop reasons were not persisted.',
              'unfinished_attempts': [r['video_id'] for r in attempts if r['video_id'] not in persisted],
              'evidence': evidence}
    write_json(dc['output'], result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('evidence', 'source_sha256')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
