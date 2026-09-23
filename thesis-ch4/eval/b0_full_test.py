"""SPEC-04-A2 one authorized run, durable records and explicit checkpoint review."""
import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time
import uuid

from data.common import config, read_jsonl, append_jsonl, write_json, write_text, fingerprint
from data.timeline import normalize_pts
from eval.generation_evidence import generate_with_evidence, stopping_evidence
from eval.reparse_v2 import IntervalParser, measurement


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def artifact(c, name, generation=False):
    root = Path(c['generation_dir'] if generation else c['measurement_dir'])
    path = root / c['files'][name]
    if not path.resolve().is_relative_to(root.resolve()):
        raise RuntimeError('Artifact path escapes designated output directory')
    return path


def read_json(p):
    return json.loads(Path(p).read_text())


def write_issue(c, slug, detail):
    p = Path(c['issue_dir']) / c['issue_name'].format(slug=slug)
    write_text(p, '# SPEC-04-A2 paused\n\n' + detail + '\n\nNo retry; persisted evidence retained.\n')
    return str(p)


def protected_paths(c):
    base = read_json(c['a1_baseline'])['protected_sha256']
    if len(base) != c['baseline_protected_count']:
        raise RuntimeError('A1 protected baseline count changed')
    paths = set(base) | set(c['protected_files'])
    # Protect all files changed by the accepted A1 delivery, not just its minimum list.
    paths.update(subprocess.check_output(['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', c['a1_commit']], text=True).splitlines())
    for pattern in c['protected_globs']:
        found = [str(p) for p in Path('.').glob(pattern) if p.is_file()]
        if not found:
            raise RuntimeError('Empty protected input glob: ' + pattern)
        paths.update(found)
    for p in paths:
        if not Path(p).is_file():
            raise RuntimeError('Missing protected input: ' + p)
        for directory in (c['generation_dir'], c['measurement_dir']):
            dest, source = Path(directory).resolve(), Path(p).resolve()
            if source == dest or source.is_relative_to(dest) or dest.is_relative_to(source):
                raise RuntimeError('Protected/output overlap: ' + p)
    return sorted(paths)


def audit(c, before=False):
    current = {p: sha(p) for p in protected_paths(c)}
    if before:
        original = read_json(c['a1_baseline'])['protected_sha256']
        if any(current[p] != v for p, v in original.items()):
            raise RuntimeError('A1 protected baseline does not match current bytes')
        for p in current:
            if p == c['spec']:
                continue
            committed = subprocess.check_output(['git', 'show', c['a1_commit'] + ':' + p])
            if hashlib.sha256(committed).hexdigest() != current[p]:
                raise RuntimeError('Accepted A1 commit mismatch: ' + p)
        result = {'created_at': now(), 'sha256': current, 'accepted_commit': c['a1_commit']}
        if artifact(c, 'before').exists():
            raise RuntimeError('Before audit already exists; refusing to reset authorization state')
        write_json(artifact(c, 'before'), result)
    else:
        baseline = read_json(artifact(c, 'before'))['sha256']
        result = {'created_at': now(), 'sha256': current, 'unchanged': current == baseline,
                  'changed': sorted(p for p in set(current) | set(baseline) if current.get(p) != baseline.get(p))}
        write_json(artifact(c, 'after'), result)
    return result


def ordered_rows(manifest, test):
    by_id = {r['video_id']: r for r in manifest}
    if len(by_id) != len(manifest) or len(set(test)) != len(test):
        raise RuntimeError('Duplicate manifest/test IDs')
    groups = {}
    for vid in sorted(test):
        row = by_id[vid]
        if row['split'] != 'test':
            raise RuntimeError('Manifest/split membership mismatch')
        groups.setdefault(row['category'], deque()).append(row)
    order = []
    while any(groups.values()):
        for category in sorted(groups):
            if groups[category]:
                order.append(groups[category].popleft())
    return order


def load_context(c):
    bc = config(c['generation_config'])
    ac = config(c['measurement_config'])
    dc = config(bc['data_config'])
    manifest = read_jsonl(dc['manifest'])
    splits = read_json(dc['splits'])
    rows = ordered_rows(manifest, splits['test'])
    if len(rows) != c['expected_test'] or len({r['category'] for r in rows[:c['warmup_n']]}) < c['warmup_min_categories']:
        raise RuntimeError('Frozen test size or warmup category coverage conflict')
    from data.make_splits import leakage
    if any(leakage(manifest, splits, dc).values()):
        raise RuntimeError('Split leakage')
    if sha(c['spec']) != c['spec_sha256']:
        raise RuntimeError('Frozen A2 body differs')
    return bc, ac, dc, rows


def initialize(c, config_path):
    bc, ac, dc, rows = load_context(c)
    for directory in (c['generation_dir'], c['measurement_dir']):
        p = Path(directory)
        existing = list(p.iterdir()) if p.exists() else []
        if directory == c['measurement_dir']:
            existing = [v for v in existing if v != artifact(c, 'tests')]
        if existing:
            raise RuntimeError('One-run output directory already populated: ' + directory)
    audit(c, before=True)
    Path(c['generation_dir']).mkdir(parents=True, exist_ok=True)
    run = {'run_id': str(uuid.uuid4()), 'created_at': now(), 'authorization': 'Author explicitly authorized exactly one full test run in the current task.',
           'spec_sha256': sha(c['spec']), 'config_sha256': sha(config_path),
           'generation_config_sha256': sha(c['generation_config']), 'generation_config': bc,
           'measurement_config_sha256': sha(c['measurement_config']),
           'source_sha256': {p: sha(p) for p in c['source_files']},
           'order_policy': 'category round-robin; categories and IDs sorted; independent of outputs',
           'order': [r['video_id'] for r in rows], 'n_test': len(rows), 'warmup_n': c['warmup_n']}
    write_json(artifact(c, 'run', True), run)
    write_text(artifact(c, 'authorization'), 'SPEC-04-A2\n\n作者在本次任务中显式授权唯一一次全 932 test 清洁运行。\n远端根目录已授权正文原样复制至 specs/；其 SHA-256 为 '+sha(c['spec'])+'。\n本地旧待授权版本未采用，未覆盖。\nrun_id='+run['run_id']+'\n')
    set_state(c, 'ready', 0, len(rows))
    print(json.dumps({'phase': 'initialized', 'run_id': run['run_id'], 'n_test': len(rows),
                      'warmup_categories': dict(Counter(r['category'] for r in rows[:c['warmup_n']]))}, ensure_ascii=False), flush=True)


def set_state(c, status, n, total, **extra):
    write_json(artifact(c, 'state'), {'status': status, 'updated_at': now(), 'n_test': total,
               'n_generated': n, 'n_interrupted': total-n, 'complete': n == total, **extra})


def evidence_valid(r, bc, c):
    try:
        if not set(c['evidence_fields']) <= set(r) or r['stop_reason'] not in c['stop_reasons']:
            return False
        ids = r['generated_token_ids']
        if not isinstance(ids, list) or any(type(v) is not int for v in ids):
            return False
        expected = stopping_evidence(ids, [], r['eos_token_id'], bc['max_new_tokens'])
        if any(r[k] != expected[k] for k in ('stop_reason', 'generated_token_count', 'max_new_tokens', 'last_token_is_eos', 'truncated')):
            return False
        if type(r['generated_token_count']) is not int or type(r['last_token_is_eos']) is not bool or type(r['truncated']) is not bool:
            return False
        return (type(r['input_prefix_token_count']) is int and r['input_prefix_token_count'] >= 0 and
                r['output_token_count'] == r['input_prefix_token_count'] + len(ids))
    except (KeyError, TypeError, ValueError):
        return False


def pause_signals(rates, c):
    return [name for name, threshold in c['monitor_thresholds'].items()
            if (rates[name] > threshold['value'] if threshold['operator'] == 'gt' else rates[name] >= threshold['value'])]


def checkpoint_path(c, n):
    return artifact(c, 'checkpoint_dir') / c['checkpoint_name'].format(n=n)


def review_path(c, n):
    return artifact(c, 'review_dir') / c['review_name'].format(n=n)


def make_checkpoint(c, records, bc, ac, rows, previous, forced=None):
    n = len(records)
    path = checkpoint_path(c, n)
    if path.exists():
        return read_json(path)
    window = records[previous:]
    errors = read_jsonl(artifact(c, 'exceptions', True))
    attempted = len(window) + len(errors)
    rates = {'generation_exception_rate': len(errors) / attempted if attempted else 0,
             'evidence_malformed_rate': sum(not evidence_valid(r, bc, c) for r in window) / len(window) if window else 0,
             'empty_raw_output_rate': sum(not str(r.get('raw_output', '')).strip() for r in window) / len(window) if window else 0,
             'truncated_rate': sum(r.get('truncated') is True for r in window) / len(window) if window else 0}
    parser = IntervalParser(ac, bc['format_regex'])
    by_id = {r['video_id']: r for r in rows}
    # Deterministic first eight in the current window, never filtered by outcomes.
    sample = [{**r, 'a1_parse': parser.parse(r['raw_output'], by_id[r['video_id']]['duration_sec'])}
              for r in window[:c['raw_sample_on_pause']]]
    cp = {'n_generated': n, 'previous_checkpoint': previous, 'window_count': len(window),
          'created_at': now(), 'rates': rates, 'raw_sample_policy': 'first configured N in the current window',
          'raw_samples': sample, 'raw_sample_sha256': fingerprint(sample), 'forced_pause': forced,
          'exceptions': errors, 'decision': 'awaiting_raw_review'}
    write_json(path, cp)
    return cp


def review(c, n, verdict, note):
    cp = read_json(checkpoint_path(c, n))
    if review_path(c, n).exists():
        raise RuntimeError('Checkpoint already reviewed; no overwriting a decision')
    if not note.strip():
        raise RuntimeError('Raw-output review note is required')
    signals = pause_signals(cp['rates'], c)
    if cp['forced_pause']:
        signals.append(cp['forced_pause'])
    if len(cp['raw_samples']) != c['raw_sample_on_pause']:
        signals.append('insufficient_raw_samples; preserve what exists, do not generate replacements')
    if verdict == 'continue' and signals:
        raise RuntimeError('Cannot override frozen implementation-error gates: ' + str(signals))
    record = {'n_generated': n, 'reviewed_at': now(), 'raw_sample_sha256': cp['raw_sample_sha256'],
              'raw_samples_reviewed': len(cp['raw_samples']), 'verdict': verdict, 'review_note': note,
              'signals_after_raw_review': signals}
    write_json(review_path(c, n), record)
    if verdict == 'pause':
        write_issue(c, 'monitor-'+str(n), json.dumps({'checkpoint': cp, 'review': record}, ensure_ascii=False, indent=2))
    print(json.dumps(record, ensure_ascii=False), flush=True)


def wait_for_review(c, cp, total):
    n = cp['n_generated']
    set_state(c, 'awaiting_review', n, total, checkpoint=str(checkpoint_path(c, n)))
    print(json.dumps({'phase': 'awaiting_review', 'n_generated': n, 'rates': cp['rates'],
                      'checkpoint': str(checkpoint_path(c, n))}, ensure_ascii=False), flush=True)
    while not review_path(c, n).exists():
        time.sleep(c['review_poll_sec'])
    r = read_json(review_path(c, n))
    if r['raw_sample_sha256'] != cp['raw_sample_sha256'] or r['verdict'] != 'continue':
        raise RuntimeError('Checkpoint paused after raw-output review')
    if pause_signals(cp['rates'], c) or cp['forced_pause']:
        raise RuntimeError('Implementation-error gate cannot be overridden')


def final_reports(c, bc, ac, rows):
    records = read_jsonl(artifact(c, 'predictions', True))
    ids = [r['video_id'] for r in records]
    by_id = {r['video_id']: r for r in rows}
    missing = [vid for vid in by_id if vid not in set(ids)]
    if len(set(ids)) != len(ids) or not set(ids) <= set(by_id):
        raise RuntimeError('Duplicate/foreign output records')
    parser = IntervalParser(ac, bc['format_regex'])
    parsed = [{**r, **parser.parse(r['raw_output'], by_id[r['video_id']]['duration_sec'])} for r in records]
    def measured(part, expected):
        result = measurement(part, by_id, dict(ac, expected_test=expected))
        result.update(stop_reason_histogram=dict(Counter(r['stop_reason'] for r in part)),
                      truncated_count=sum(r.get('truncated') is True for r in part),
                      truncated_rate=sum(r.get('truncated') is True for r in part)/len(part) if part else None,
                      evidence_malformed_count=sum(not evidence_valid(r, bc, c) for r in part))
        return result
    metrics = {'spec': 'SPEC-04-A2', 'n_test': len(rows), 'n_generated': len(records),
               'n_interrupted': len(missing), 'interrupted': missing, 'complete': not missing,
               'overall': measured(parsed, len(rows)),
               'by_category': {category: measured([r for r in parsed if by_id[r['video_id']]['category'] == category],
                                                    sum(r['category'] == category for r in rows))
                               for category in sorted({r['category'] for r in rows})}}
    write_json(artifact(c, 'parsed'), {'spec': 'SPEC-04-A2', 'records': parsed})
    write_json(artifact(c, 'metrics'), metrics)
    old = read_jsonl(c['old_predictions'])
    new = {r['video_id']: r for r in records}
    overlap = [r for r in old if r['video_id'] in new]
    differences = [{'video_id': r['video_id'], 'old_raw_output': r['raw_output'],
                    'new_raw_output': new[r['video_id']]['raw_output']}
                   for r in overlap if r['raw_output'].encode('utf-8') != new[r['video_id']]['raw_output'].encode('utf-8')]
    repro = {'expected_overlap': len(old), 'compared_count': len(overlap), 'exact_count': len(overlap)-len(differences),
             'reproduced_exact_rate': (len(overlap)-len(differences))/len(overlap) if overlap else None,
             'comparison_complete': len(overlap) == len(old) == c['expected_old_overlap'],
             'differences': differences, 'diagnostic_only': True, 'not_a_pass_fail_threshold': True}
    write_json(artifact(c, 'reproducibility'), repro)
    after = audit(c)
    attempts = read_jsonl(artifact(c, 'attempts', True))
    attempted_ids = [r['video_id'] for r in attempts]
    full = len(records) == len(rows) == c['expected_test'] and not missing
    checkpoints = sorted(artifact(c, 'checkpoint_dir').glob('*.json'))
    reviews = [read_json(review_path(c, read_json(p)['n_generated'])) for p in checkpoints if review_path(c, read_json(p)['n_generated']).exists()]
    monitor_complete = bool(checkpoints) and len(reviews) == len(checkpoints) and all(r['verdict'] == 'continue' and r['raw_samples_reviewed'] == c['raw_sample_on_pause'] for r in reviews)
    checks = [
        ('8.1', '覆盖等于 932 才 complete', full, {'coverage': len(records), 'interrupted': missing}),
        ('8.2', 'protected before/after unchanged', after['unchanged'], {'count': len(after['sha256']), 'changed': after['changed']}),
        ('8.3', '全 932 权威停止证据与判定序', full and all(evidence_valid(r, bc, c) for r in records), {'stop_reasons': metrics['overall']['stop_reason_histogram'], 'malformed': metrics['overall']['evidence_malformed_count']}),
        ('8.4', '312 重叠视频可复现性诊断已产出', repro['comparison_complete'], {'compared': len(overlap), 'rate': repro['reproduced_exact_rate'], 'rate_is_not_a_gate': True}),
        ('8.5', '账本 generated + interrupted = n_test = 932 且无重复调用', len(records)+len(missing) == len(rows) == c['expected_test'] and len(set(attempted_ids)) == len(attempted_ids) and set(ids) <= set(attempted_ids) <= set(by_id), {'n_test': len(rows), 'n_generated': len(records), 'n_interrupted': len(missing), 'attempted': len(attempts)}),
        ('8.6', '新产物只写新目录，旧产物逐字节不变', after['unchanged'], {'generation_dir': c['generation_dir'], 'measurement_dir': c['measurement_dir']})]
    accept = {'spec': 'SPEC-04-A2', 'checks': [{'id': i, 'criterion': label, 'passed': bool(ok), 'evidence': e} for i,label,ok,e in checks],
              'complete': full, 'n_test': len(rows), 'n_generated': len(records), 'n_interrupted': len(missing),
              'interrupted': missing, 'monitor_reviews_complete': monitor_complete,
              'generation_config_unchanged': sha(c['generation_config']) == read_json(artifact(c, 'run', True))['generation_config_sha256'],
              'status': 'passed' if all(check[2] for check in checks) and monitor_complete else 'incomplete_or_paused',
              'issues': sorted(str(p) for p in Path(c['issue_dir']).glob('04a2-*.md'))}
    write_json(artifact(c, 'acceptance'), accept)
    write_text(artifact(c, 'report'), '# SPEC-04-A2 全 test B0\n\n'+json.dumps(metrics,ensure_ascii=False,indent=2)+'\n\n可复现性诊断：\n'+json.dumps({k:v for k,v in repro.items() if k != 'differences'},ensure_ascii=False,indent=2)+'\n\n不一致文本详见 reproducibility.json；不作为验收失败门。\n')
    return accept


def worker(c, config_path, resume=False):
    bc, ac, dc, rows = load_context(c)
    run = read_json(artifact(c, 'run', True))
    if run['config_sha256'] != sha(config_path) or run['generation_config'] != bc or run['order'] != [r['video_id'] for r in rows] or any(sha(p) != v for p,v in run['source_sha256'].items()):
        raise RuntimeError('Run configuration/order/source changed; no generation permitted')
    if not audit(c)['unchanged']:
        raise RuntimeError('Protected files changed before generation')
    lock_path = artifact(c, 'worker_lock', True)
    with open(lock_path, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if artifact(c, 'launch', True).exists() and not resume:
            raise RuntimeError('Authorization already launched; only resume of unstarted items is permitted')
        if not artifact(c, 'launch', True).exists():
            if resume:
                raise RuntimeError('Cannot resume before first launch')
            write_json(artifact(c, 'launch', True), {'run_id': run['run_id'], 'started_at': now()})
        records = read_jsonl(artifact(c, 'predictions', True))
        attempts = read_jsonl(artifact(c, 'attempts', True))
        done = {r['video_id'] for r in records}
        started = {r['video_id'] for r in attempts}
        if len(done) != len(records) or len(started) != len(attempts) or not done <= started or not started <= set(run['order']):
            raise RuntimeError('Invalid run ledger')
        model = tokenizer = None
        if started != done:
            raise RuntimeError('In-flight attempt has no durable result; cannot infer safely again: '+str(sorted(started-done)))
        if read_jsonl(artifact(c, 'exceptions', True)):
            raise RuntimeError('Prior generation exception; another authorization is required for repairs/retry')
        timeline = {r['video_id']: r for r in read_jsonl(dc['timeline_records'])}
        previous = 0
        for p in sorted(artifact(c, 'review_dir').glob('*.json')):
            reviewed = read_json(p)
            if reviewed['verdict'] != 'continue':
                raise RuntimeError('An implementation-error review stopped this run; no automatic restart')
            previous = max(previous, reviewed['n_generated'])
        target = min(previous+c['checkpoint_every'] if previous else c['warmup_n'], len(rows))
        try:
            while True:
                if len(records) >= target:
                    if previous == len(rows):
                        break
                    cp = make_checkpoint(c, records, bc, ac, rows, previous)
                    wait_for_review(c, cp, len(rows))
                    previous = len(records)
                    if previous == len(rows):
                        break
                    target = min(previous+c['checkpoint_every'], len(rows))
                if model is None:
                    import torch
                    from src.model.holmesvau_infer import load_model
                    torch.set_num_threads(bc['torch_threads'])
                    model, tokenizer = load_model(bc['model_path'], bc['device'])
                from decord import VideoReader, cpu
                from src.model.holmesvau_infer import uniform_indices
                set_state(c, 'running', len(records), len(rows), next_checkpoint=target)
                for row in rows:
                    vid = row['video_id']
                    if vid in done:
                        continue
                    if len(records) >= target:
                        break
                    phase = 'input_validation'
                    try:
                        times = normalize_pts(timeline[vid]['pts'], timeline[vid]['time_base'])
                        reader = VideoReader(row['path'], ctx=cpu(dc['stream_index']), num_threads=dc['decode_threads'])
                        if len(reader) != len(times):
                            raise RuntimeError('Decoder frame count differs from normalized timeline')
                        indices = uniform_indices(len(reader), bc['num_frames'])
                        del reader
                        frame_times = bc['frame_time_separator'].join(bc['frame_time_template'].format(index=i, time_sec=float(times[j])) for i,j in enumerate(indices,start=1))
                        prompt = bc['prompt_template'].format(category=row['category'], duration_sec=row['duration_sec'], frame_times=frame_times)
                        attempt = {'video_id': vid, 'category': row['category'], 'started_at': now(), 'prompt': prompt, 'frame_indices': indices,
                                   'generation_kwargs': {k: bc[k] for k in c['generation_keys']}, 'run_id': run['run_id']}
                        append_jsonl(artifact(c, 'attempts', True), attempt)
                        attempts.append(attempt)
                        phase = 'generate_and_evidence'
                        begin = time.monotonic()
                        raw, actual_indices, evidence = generate_with_evidence(row['path'], prompt, model, tokenizer, **attempt['generation_kwargs'])
                        # Generation stage stores raw text and authoritative evidence only.
                        result = {'video_id': vid, 'category': row['category'], 'raw_output': raw, 'frames_seen': len(actual_indices),
                                  'frame_indices': actual_indices, 'finished_at': now(), 'elapsed_sec': time.monotonic()-begin,
                                  'run_id': run['run_id'], **evidence}
                        append_jsonl(artifact(c, 'predictions', True), result)
                        records.append(result); done.add(vid)
                        if actual_indices != indices:
                            raise RuntimeError('Returned frame indices differ from prompt timestamps')
                        if not evidence_valid(result, bc, c):
                            cp = make_checkpoint(c, records, bc, ac, rows, previous, forced='evidence_malformed')
                            wait_for_review(c, cp, len(rows))
                        set_state(c, 'running', len(records), len(rows), next_checkpoint=target)
                        print(json.dumps({'phase': 'generated', 'n': len(records), 'category': row['category'], 'stop_reason': evidence['stop_reason'],
                                          'tokens': evidence['generated_token_count'], 'elapsed_sec': result['elapsed_sec']},ensure_ascii=False),flush=True)
                    except Exception as exc:
                        append_jsonl(artifact(c, 'exceptions', True), {'video_id': vid, 'phase': phase, 'error': repr(exc), 'at': now()})
                        cp = make_checkpoint(c, records, bc, ac, rows, previous, forced='generation_or_pipeline_exception')
                        wait_for_review(c, cp, len(rows))
                        raise
            acceptance = final_reports(c, bc, ac, rows)
            set_state(c, 'complete' if acceptance['status'] == 'passed' else 'acceptance_failed', len(records), len(rows), authorization_consumed=True)
            print(json.dumps({'phase': 'finished', 'acceptance': acceptance['status'], 'n_generated': len(records)}, ensure_ascii=False), flush=True)
        except BaseException as exc:
            write_issue(c, 'run-paused', repr(exc))
            final_reports(c, bc, ac, rows)
            set_state(c, 'paused', len(records), len(rows), error=repr(exc))
            raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['init', 'run', 'review', 'report'])
    p.add_argument('--config', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--checkpoint', type=int)
    p.add_argument('--verdict', choices=['continue', 'pause'])
    p.add_argument('--note')
    args = p.parse_args()
    c = config(args.config)
    try:
        if args.command == 'init':
            initialize(c, args.config)
        elif args.command == 'run':
            worker(c, args.config, args.resume)
        elif args.command == 'review':
            review(c, args.checkpoint, args.verdict, args.note or '')
        else:
            bc, ac, dc, rows = load_context(c)
            print(json.dumps(final_reports(c, bc, ac, rows), ensure_ascii=False))
    except Exception as exc:
        write_issue(c, 'command-'+args.command, repr(exc))
        raise


if __name__ == '__main__':
    main()
